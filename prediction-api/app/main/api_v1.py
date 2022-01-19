from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request

from app.main.prediction_service import get_prediction_result, execute_arima, execute_snaive, execute_naive, \
    execute_lstm

api_v1 = Blueprint('api_v1', __name__)

executor = ThreadPoolExecutor(max_workers=10)


@api_v1.route('/')
def index():
    return 'This is my (Aikaterini Mitsopoulou) thesis - Prediction API'


@api_v1.route('/prediction')
def prediction():
    prediction_id = request.args.get('id')
    return get_prediction_result(prediction_id)


@api_v1.route('/naive', methods=['POST'])
def naive():
    uploaded_file = request.files['file_upload']
    filename = uploaded_file.filename
    uploaded_file.save(uploaded_file.filename)

    train_year = request.form.get('train_year')
    test_year = request.form.get('test_year')
    id = f"{uploaded_file.filename}-naive-{train_year}-{test_year}"

    executor.submit(execute_naive, filename, train_year, test_year, id)

    return {'status': 'submitted', 'id': id}


@api_v1.route('/snaive', methods=['POST'])
def snaive():
    uploaded_file = request.files['file_upload']
    filename = uploaded_file.filename
    uploaded_file.save(uploaded_file.filename)

    train_year = request.form.get('train_year')
    test_year = request.form.get('test_year')
    id = f"{uploaded_file.filename}-snaive-{train_year}-{test_year}"

    executor.submit(execute_snaive, filename, train_year, test_year, id)

    return {'status': 'submitted', 'id': id}


@api_v1.route('/arima', methods=['POST'])
def arima():
    uploaded_file = request.files['file_upload']
    filename = uploaded_file.filename
    uploaded_file.save(uploaded_file.filename)

    train_year = request.form.get('train_year')
    test_year = request.form.get('test_year')
    id = f"{uploaded_file.filename}-arima-{train_year}-{test_year}"

    executor.submit(execute_arima, filename, train_year, test_year, id)

    return {'status': 'submitted', 'id': id}


@api_v1.route('/lstm', methods=['POST'])
def lstm():
    uploaded_file = request.files['file_upload']
    filename = uploaded_file.filename
    uploaded_file.save(uploaded_file.filename)

    train_year = request.form.get('train_year')
    test_year = request.form.get('test_year')
    id = f"{uploaded_file.filename}-lstm-{train_year}-{test_year}"

    executor.submit(execute_lstm, filename, id)

    return {'status': 'submitted', 'id': id}