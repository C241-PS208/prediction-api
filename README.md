
# ShaveMax Prediction API

This Flask application is specific application that handles the prediction API of the ShaveMax Application.

In this Application, there is 3 endpoints:
1. "/": an endpoint with GET method to return all available endpoints of the application.
2. "/predict": an endpoint with POST method that requires 
request body gender (text) and image(file). This endpoint will return the face shape, hair type, and hairstyle recommendations as json.
3. "/status": an endpoint with GET methodthat returns the status of current application.

This application is deployed via Google Cloud Compute Engines and all the models needed is uploaded to the compute engine. Then, the applicatin will be running on the compute engine and kept alive by pm2 process manager.

## Replication and How to run the application
1. Clone the repository: https://github.com/C241-PS208/prediction-api.git
2. Create virtual env: “python -m venv venv” (Do this, when cloning project for the first time, otherwise skip to step 3)
3. Activate virtual env: “venv\Scripts\activate” or “source venv\Scripts\activate”
4. Install the requirements of library: “pip install -r requirements.txt” (Do this, when cloning project for the first time, otherwise skip to step 5)
5. Run the application: “flask --app main.py --debug run”



