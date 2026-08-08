from collections.abc import Callable
from cProfile import Profile
from pathlib import Path
from pstats import Stats

from faker import Faker
from sentence_similarity import sentence_similarity


def profile(function: Callable) -> None:
    """Profile a function."""
    with Profile() as profiler:
        function()
        Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(20)


def fake_sentences(n: int = 3000) -> None:
    """Create fake sentences and store these into 'fake.txt'."""
    fake = Faker()
    words = fake.words(80)
    sentences = [
        fake.sentence(15, variable_nb_words=True, ext_word_list=words)
        for _ in range(n)
    ]
    with Path("fake.txt").open("w") as file:
        file.write("\n".join(sentences))


def benchmark() -> None:
    """Run a benchmark on 'fake.csv'."""
    sentences = Path("fake.txt").read_text().splitlines()

    def run() -> None:
        sentence_similarity(sentences)

    profile(run)


def main() -> None:
    """Main runner."""
    fake_sentences(3000)
    benchmark()


if __name__ == "__main__":
    import subprocess

    subprocess.run("clear", check=True, shell=True)
    main()
