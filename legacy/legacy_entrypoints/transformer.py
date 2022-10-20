from __future__ import annotations

import json
import typing as t
from pathlib import Path

from IPython.display import display
from pydantic import BaseModel, Field
from torch_iter import FixedSubsetIterPlannerBuilder, FracSubsetSize
from torch_iter.metric import SmallerIsBetterCriteria
from tqdm.notebook import tqdm
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.utils.generic import PaddingStrategy

from fp_ell.data import Collator
from fp_ell.environment import EnvironmentType
from fp_ell.fp16 import FP16Config
from fp_ell.fs import create_fs_for_env_type
from fp_ell.gpu import is_gpu_device, ensure_gpu_model
from fp_ell.logging import LoggerBuilderModel
from fp_ell.loss import LossModel, MSELossModel
from fp_ell.lr_scheduling import CosineScheduleWithWarmupLRSchedulerBuilder
from fp_ell.model import AutoModelBuilder, PoolerType
from fp_ell.optim import NakamaOptimizerBuilder
from fp_ell.seed import seed_everything
from fp_ell.tokenization import get_tokenizer_for_backbone
from fp_ell.training import train_k_fold_models
from fp_ell.typing import read_csv, pd_series
from fp_ell.validation import ValidationStrategy, analyze_val_strategy, build_fold_result_df

_TARGET_LIST = [
    'cohesion',
    'syntax',
    'vocabulary',
    'phraseology',
    'grammar',
    'conventions',
]


class TrainTransformerParams(BaseModel):
    run_id: str
    fold_list: t.List[int]

    env_type: EnvironmentType = EnvironmentType.colab
    seed: int = 42
    device: str = 'cuda'
    num_folds: int = 5
    expected_gpu_model: str = 'tesla_p100_pcie_16gb'
    write_model_to_disk: bool = False
    logger_list: t.List[LoggerBuilderModel] = Field(default_factory=list)

    backbone: str = 'microsoft/deberta-v3-base'
    loss: LossModel = Field(default_factory=MSELossModel)
    pooler_type: PoolerType = PoolerType.mean
    max_len: int = 1430
    batch_size: int = 2
    accumulate_gradient_steps: int = 4
    encoder_lr: float = 2e-5
    decoder_lr: float = 2e-5
    num_epochs: int = 4
    warmup_steps_ratio: float = 0.0
    num_cycles: float = 0.5
    val_freq: float = 0.5
    fp16: t.Optional[FP16Config] = None

    @classmethod
    def from_file(cls, file_path: Path) -> TrainTransformerParams:
        with open(file_path) as f:
            return TrainTransformerParams(**json.load(f))


def train_transformer_model(params: TrainTransformerParams):
    if is_gpu_device(params.device):
        ensure_gpu_model(params.expected_gpu_model)

    fs = create_fs_for_env_type(env_type=params.env_type)
    fs.initialize()

    # Derived parameters
    backbone_escaped = params.backbone.replace('/', '-')
    run_base_name = f'{backbone_escaped}-{params.run_id}'
    run_name_template = f'{run_base_name}-fold_{{fold}}-seed_{params.seed}'
    oof_path_template = fs.oof_dir / f'{run_name_template}.csv'

    # Set seed
    seed_everything(params.seed)

    # Init tqdm for pandas
    tqdm.pandas()

    # Data loading
    all_df = read_csv(fs.dataset_dir / 'train.csv')
    print('Original CSV:')
    display(all_df)
    cv_strategy = ValidationStrategy(num_folds=params.num_folds, target_list=_TARGET_LIST, num_bins=5)
    all_df = cv_strategy.assign_folds(all_df)
    tokenizer = get_tokenizer_for_backbone(params.backbone, padding_strategy=PaddingStrategy.DO_NOT_PAD)
    all_df['full_text_tok_len'] = pd_series(all_df['full_text'])\
        .progress_apply(lambda text: len(tokenizer.tokenize(text, max_len=params.max_len)))
    print('CV split:')
    display(analyze_val_strategy(df=all_df, strategy=cv_strategy, target_list=_TARGET_LIST))

    score_list = train_k_fold_models(
        all_df=all_df,
        fold_list=params.fold_list,
        target_list=_TARGET_LIST,
        model_builder=AutoModelBuilder(
            params.backbone,
            num_targets=len(_TARGET_LIST),
            pooler_type=params.pooler_type),
        tokenizer=tokenizer,
        criterion=params.loss.get_inner(),
        train_iter_planner_builder=FixedSubsetIterPlannerBuilder(subset_size=FracSubsetSize(params.val_freq)),
        device=params.device,
        to_checkpoint_template=str(fs.models_dir / f'{run_name_template}.pt'),
        num_epochs=params.num_epochs,
        optimizer_builder=NakamaOptimizerBuilder(
            weight_decay=1e-2,
            encoder_lr=params.encoder_lr,
            decoder_lr=params.decoder_lr,
            eps=1e-6,
            betas=(0.9, 0.999)),
        lr_scheduler_builder=CosineScheduleWithWarmupLRSchedulerBuilder(
            warmup_steps_ratio=params.warmup_steps_ratio,
            num_cycles=params.num_cycles),
        batch_size=params.batch_size,
        max_len=params.max_len,
        grad_scaler=params.fp16.to_grad_scaler() if params.fp16 is not None else None,
        num_workers=2,
        oof_pred_path_template=oof_path_template,
        model_comparison_metric='valid_mcrmse',
        model_comparison_metric_criteria=SmallerIsBetterCriteria(),
        collator=Collator(DataCollatorWithPadding(tokenizer.tokenizer)),
        accumulate_gradient_steps=params.accumulate_gradient_steps,
        write_model_to_disk=params.write_model_to_disk,
        logger_builder_list=[lm.get_inner() for lm in params.logger_list])

    print('Results:')
    display(build_fold_result_df(params.fold_list, score_list))
