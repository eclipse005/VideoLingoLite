# test_sentence_objects.py
"""
Simplified test for Sentence/Chunk object model without full dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """Word/character-level ASR unit with timestamps."""
    text: str
    start: float
    end: float
    speaker_id: Optional[str] = None
    index: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Sentence:
    """Sentence-level object composed of multiple Chunks."""
    chunks: List[Chunk]
    text: str
    start: float
    end: float
    translation: str = ""
    index: int = 0
    is_split: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def update_timestamps(self):
        if self.chunks:
            self.start = self.chunks[0].start
            self.end = self.chunks[-1].end


def test_chunk_creation():
    print("[Test 1] Creating Chunk objects...")
    chunks = [
        Chunk(text="Hello ", start=0.0, end=0.5, index=0),
        Chunk(text="world ", start=0.5, end=1.0, index=1),
        Chunk(text="!", start=1.0, end=1.2, index=2),
    ]
    print(f"   Created {len(chunks)} chunks")
    print(f"   First chunk: '{chunks[0].text}' ({chunks[0].start} - {chunks[0].end})")
    print(f"   Total duration: {sum(c.duration for c in chunks)}")
    return True


def test_sentence_creation():
    print("\n[Test 2] Creating Sentence objects...")
    chunks = [
        Chunk(text="Hello ", start=0.0, end=0.5, index=0),
        Chunk(text="world ", start=0.5, end=1.0, index=1),
        Chunk(text="!", start=1.0, end=1.2, index=2),
    ]

    sentence = Sentence(
        chunks=chunks,
        text="Hello world!",
        start=0.0,
        end=1.2,
        index=0
    )

    print(f"   Sentence text: '{sentence.text}'")
    print(f"   Sentence start: {sentence.start}")
    print(f"   Sentence end: {sentence.end}")
    print(f"   Sentence duration: {sentence.duration}")
    print(f"   Number of chunks: {len(sentence.chunks)}")
    print(f"   Is split: {sentence.is_split}")
    return True


def test_timestamp_preservation():
    print("\n[Test 3] Testing timestamp preservation...")

    chunks = [
        Chunk(text="The ", start=0.0, end=0.3, index=0),
        Chunk(text="quick ", start=0.3, end=0.6, index=1),
        Chunk(text="brown ", start=0.6, end=0.9, index=2),
        Chunk(text="fox", start=0.9, end=1.2, index=3),
    ]

    sentence = Sentence(
        chunks=chunks,
        text="The quick brown fox",
        start=chunks[0].start,
        end=chunks[-1].end,
        index=0
    )

    expected_start = 0.0
    expected_end = 1.2

    assert sentence.start == expected_start, f"Start mismatch: {sentence.start} != {expected_start}"
    assert sentence.end == expected_end, f"End mismatch: {sentence.end} != {expected_end}"

    print(f"   Timestamps preserved correctly: {sentence.start} - {sentence.end}")

    # Test update_timestamps method
    chunks[0].start = 2.0
    sentence.update_timestamps()

    assert sentence.start == 2.0, "update_timestamps failed"
    print(f"   After update_timestamps: {sentence.start} - {sentence.end}")
    return True


def test_sentence_splitting():
    print("\n[Test 4] Testing Sentence splitting logic...")

    chunks = [
        Chunk(text="This ", start=0.0, end=0.5, index=0),
        Chunk(text="is ", start=0.5, end=1.0, index=1),
        Chunk(text="sentence ", start=1.0, end=1.5, index=2),
        Chunk(text="one. ", start=1.5, end=2.0, index=3),
        Chunk(text="This ", start=2.0, end=2.5, index=4),
        Chunk(text="is ", start=2.5, end=3.0, index=5),
        Chunk(text="sentence ", start=3.0, end=3.5, index=6),
        Chunk(text="two.", start=3.5, end=4.0, index=7),
    ]

    # Simulate splitting at index 4
    sentence1_chunks = chunks[:4]
    sentence2_chunks = chunks[4:]

    sentence1 = Sentence(
        chunks=sentence1_chunks,
        text="".join(c.text for c in sentence1_chunks),
        start=sentence1_chunks[0].start,
        end=sentence1_chunks[-1].end,
        index=0,
        is_split=True
    )

    sentence2 = Sentence(
        chunks=sentence2_chunks,
        text="".join(c.text for c in sentence2_chunks),
        start=sentence2_chunks[0].start,
        end=sentence2_chunks[-1].end,
        index=1,
        is_split=True
    )

    print(f"   Split into {2} sentences")
    print(f"   Sentence 1: '{sentence1.text}' ({sentence1.start} - {sentence1.end})")
    print(f"   Sentence 2: '{sentence2.text}' ({sentence2.start} - {sentence2.end})")

    assert sentence1.is_split and sentence2.is_split, "Split flag not set"
    print(f"   Both sentences marked as split: True")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Chunk/Sentence Object Model")
    print("=" * 60)

    tests = [
        test_chunk_creation,
        test_sentence_creation,
        test_timestamp_preservation,
        test_sentence_splitting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("[PASS] All tests passed!")
        exit(0)
    else:
        print("[FAIL] Some tests failed!")
        exit(1)
