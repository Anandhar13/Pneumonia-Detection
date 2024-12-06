import pytest
from pneumonia_detection_model import PneumoniaDetectionModel

@pytest.fixture
def pneumonia_model():

    return PneumoniaDetectionModel(input_shape=(256, 256, 3), learning_rate=0.001)

def test_model_initialization(pneumonia_model):

    # Test to check that the model is initialized properly.

    # Ensure the model is not None
    assert pneumonia_model.model is not None, "The model was not initialized correctly."

def test_model_layers_count(pneumonia_model):

    # Test to check the number of layers in the model.

    model = pneumonia_model.model

    # Check if the model has more than 5 layers (ensures the architecture is built)
    assert len(model.layers) > 5, "The model does not have enough layers."

def test_model_training_compilation(pneumonia_model):

    # Test to ensure the model is compiled correctly.

    model = pneumonia_model.model

    # Check if the optimizer and loss are set
    assert model.optimizer is not None, "The model is not compiled with an optimizer."
    assert model.loss == 'binary_crossentropy', "The loss function is not binary_crossentropy."
