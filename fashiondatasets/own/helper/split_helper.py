from pathlib import Path

from fashionscrapper.utils.io import Json_DB
from sklearn.model_selection import train_test_split


class SplitLoader:
    def __init__(self, base_path):
        self.entries_path = Path(base_path, "entries.json")

        self.split_entries_path = {
            "test": Path(base_path, "entries_test.json"),
            "train": Path(base_path, "entries_train.json"),
            "validation": Path(base_path, "entries_validation.json")
        }

    def build_entries_splits(self):
        with Json_DB(self.entries_path) as entries_db:
            entries = (entries_db.all())
            train, rest = train_test_split(entries, test_size=0.3)
            validation, test = train_test_split(rest, test_size=0.5)

            splits = {
                "train": train, "validation": validation, "test": test
            }

            for split, split_data in splits.items():
                split_path = self.split_entries_path[split]

                if split_path.exists():
                    split_path.unlink()

                with Json_DB(split_path) as split_db:
                    split_db.insert_multiple(split_data)

    def load_entries(self, splits=None):
        if not (all([x.exists() for x in self.split_entries_path.values()])):
            self.build_entries_splits()

        if not splits:
            splits = self.split_entries_path.keys()

        d = {}
        for split in splits:
            split_path = self.split_entries_path[split]

            with Json_DB(split_path) as split_db:
                d[split] = split_db.all()
        return d