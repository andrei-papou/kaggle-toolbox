import csv
import typing as t
from pathlib import Path


class OOFPredDict(t.Dict[str, t.List[float]]):

    def save_to_csv(self, csv_path: Path, score_col_name_list: t.List[str]):
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['id'] + score_col_name_list)
            writer.writeheader()
            for id, score_list in self.items():
                writer.writerow({
                    'id': id,
                    **dict(zip(score_col_name_list, score_list)),
                })
