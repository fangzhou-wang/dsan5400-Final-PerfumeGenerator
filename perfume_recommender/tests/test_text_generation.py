from perfume_recommender.text_generation.text_generation import TextGenerator
import pytest


@pytest.fixture
def text_generator():
    return TextGenerator(model_path="gpt2")


def test_generate_description(text_generator):
    description = text_generator.generate_description(
        "This perfume is floral and woody"
    )
    assert isinstance(description, str), "Generated description should be a string"
    assert len(description) > 0, "Generated description should not be empty"
