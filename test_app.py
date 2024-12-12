import pytest
from app import PneumoniaDetector

@pytest.fixture
def client():
    # Set up a test client for the Flask App.
    model_path = 'models/pnemonia_model.h5'
    app_instance = PneumoniaDetector(model_path)
    app_instance.app.config['TESTING'] = True  # Enable Flask test mode
    client = app_instance.app.test_client()
    return client

def test_home_page(client):
    # Test that the home page loads successfully.
    response = client.get('/')  # Send a GET request to the home page
    assert response.status_code == 200  # The home page should load successfully
    assert b'<form' in response.data  # Check for a form element in the HTML

def test_predict_route_without_image(client):
    # Test that the predict route handles a POST request without an image.
    response = client.post('/', data={})  # Send a POST request without data
    assert response.status_code == 400  # Expecting a 400 Bad Request or appropriate error handling
    # Optionally, check for an error message in the response
    # assert b'No image file uploaded' in response.data
