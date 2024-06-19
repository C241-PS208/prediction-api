
# ShaveMax Prediction API

## Description
ShaveMax is a mobile application that utilized the machine learning technology to predict the face shape and hair type of the user based on the photo provided to the system and recomment suitable hairstyles for the users.

## Meet Our Team
| ID             | Learning Path       | University                          | Name                       | Status |
|----------------|---------------------|-------------------------------------|----------------------------|--------|
| M009D4KY2095   | Machine Learning    | Gunadarma University                | Gilang Ferdiansyah         | Active |
| M009D4KX2419   | Machine Learning    | Gunadarma University                | Nadira Putri Bimono        | Active |
| M009D4KY1905   | Machine Learning    | Gunadarma University                | Josep Samuel Angelo        | Active |
| C010D4KY1226   | Cloud Computing     | Universitas Indonesia               | Vinsensius Ferdinando      | Active |
| C010D4KY0957   | Cloud Computing     | Universitas Indonesia               | Bintang Pratama            | Active |
| A010D4KY3439   | Mobile Development  | Universitas Indonesia               | Rama Tridigdaya            | Active |
| A550D4NY4608   | Mobile Development  | UIN Syarif Hidayatullah Jakarta     | Muhammad Aryodiro Sunaryo  | Active |


## Functionality
This Flask application is specific application that handles the prediction API of the ShaveMax Application.

## In this Application, there is 3 endpoints:
#### Get all available endpoints
```http
  GET /api/items
```
An endpoint with GET method to return all available endpoints of the application.
#### Get item
```http
  POST /predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `image` | `file` | **Required**. Image of the face with hair |
| `gender` | `string` | **Required**. Gender of the User in the image|

An endpoint with POST method that requires 
request body gender (text) and image(file). This endpoint will return the face shape, hair type, and hairstyle recommendations as json.

#### Get item
```http
  GET /status
```
An endpoint with GET methodthat returns the status of current application.

This application is deployed via Google Cloud Compute Engines and all the models needed is uploaded to the compute engine. Then, the applicatin will be running on the compute engine and kept alive by pm2 process manager.

## Replication and How to Run the Application

Follow these steps to set up and run the application:

### 1. Clone the Repository
First, clone the repository to your local machine using the following command:
```sh
git clone https://github.com/C241-PS208/prediction-api.git
```
### 2. Create virtual environment
Second, create the virtual environment (Do this only when cloning project for the first time, otherwise skip to step 3):
```sh
python -m venv venv
```
### 3. Activate virtual environment
Third, activate the virtual env with the script below:
```sh
venv\Scripts\activate
```
or
```sh
source venv\Scripts\activate
```
### 4. Install the requirements of library
Fourth, install the library requirements(Do this only when cloning project for the first time, otherwise skip to step 5):
```sh
git clone https://github.com/C241-PS208/prediction-api.git
```
### 5. Run the application
Last, run the application with this script:
```sh
flask --app main.py --debug run
```



