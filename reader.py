import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "original_survey_df.pkl",
)


@dataclass
class SciReviewSample:
    paper_id: str
    title: str
    abstract: str
    sections: list
    texts: list
    n_bibs: list
    n_nonbibs: list
    bib_titles: list
    bib_abstracts: list
    bib_citing_sentences: list
    split: str


def load_scireviewgen(
    dataset_path: str = DEFAULT_DATA_PATH,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
) -> List[SciReviewSample]:
    """
    Load SciReviewGen dataset and return a list of SciReviewSample objects.

    Args:
        dataset_path: Path to original_survey_df.pkl.
        num_samples: Optional number of samples to keep (takes head after optional shuffle).
        shuffle: Shuffle before trimming to num_samples.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_pickle(dataset_path)
    total = len(df)

    if shuffle:
        df = df.sample(frac=1, random_state=42)
    if num_samples is not None:
        df = df.head(num_samples)

    samples: List[SciReviewSample] = []
    for paper_id, row in df.iterrows():
        samples.append(
            SciReviewSample(
                paper_id=paper_id,
                title=row["title"],
                abstract=row["abstract"],
                sections=row["section"],
                texts=row["text"],
                n_bibs=row["n_bibs"],
                n_nonbibs=row["n_nonbibs"],
                bib_titles=row["bib_titles"],
                bib_abstracts=row["bib_abstracts"],
                bib_citing_sentences=row["bib_citing_sentences"],
                split=row["split"],
            )
        )

    print(f"Loaded {len(samples)} / {total} papers from {dataset_path}")
    print(f"Columns: {list(df.columns)}")
    if "split" in df.columns:
        print(f"Split distribution: {df['split'].value_counts().to_dict()}")
    if "section" in df.columns:
        avg_sections = df["section"].apply(len).mean()
        print(f"Average #sections per paper: {avg_sections:.2f}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="SciReviewGen reader")
    parser.add_argument(
        "--dataset_path",
        default=DEFAULT_DATA_PATH,
        help="Path to original_survey_df.pkl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to read (after optional shuffle)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before sampling",
    )
    args = parser.parse_args()

    samples = load_scireviewgen(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
    )

    # Simple preview
    preview_n = min(2, len(samples))
    for sample in samples[:preview_n]:
        print("-" * 60)
        print(f"{sample.paper_id} | split={sample.split}")
        print(f"title: {sample.title[:120]}")
        print(f"#sections: {len(sample.sections)}, #text: {len(sample.texts)}")
        print(f"#first section: \n{sample.sections[0]}\n{sample.texts[0][:500]}")
    print("Done.")


if __name__ == "__main__":
    main()

