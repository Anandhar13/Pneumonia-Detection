import pytest
from pneumonia_detection_model import PneumoniaDetection


@pytest.fixture
def model():

    #creates a instance of PneumoniaDetection model

    return PneumoniaDetection()

def test_pneumonia_model(model):
    assert model.input_shape == (256,256,256) #checks if the input shape is in the given dimension
    assert model.learning_rate == 0.001 #learning rate should be 0.001

def test_model_compilation(model):
    #test the model compilation

    assert model.model.optimizer.get_config()["name"] == "Adam" #Optimizer sohould be adam
    assert model.model.loss == "binary_crossentropy" #should be the loss function
    assert "accuracy" in model.model.metrics_names , "Metrics should be included in the accuray"

def test_model_structure(model):
    #The model has 5 Conv2D layers, 1 Flatten, 2 Dense, and other utility layers
    total_layers = sum(1 for layers in model.model.layers)
    assert total_layers == 13, f"Model should have 12 layers, but found {total_layers}"
