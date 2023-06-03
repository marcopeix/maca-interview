import unittest
import pandas as pd
from io import StringIO
from main import read_sellers_data, get_closed_deals, get_closed_deals_per_origin, compute_average_close_time, compute_avg_declared_monthly_revenue

class TestReadSellersData(unittest.TestCase):

    def setUp(self):
        # Set up a mock response using test data
        test_data = '''mql_id,first_contact_date
                      1,2021-01-01
                      2,2021-02-01
                      3,2021-03-01'''
        self.mock_response = StringIO(test_data)

        # Patch the requests.get function to return the mock response
        self.patcher = unittest.mock.patch('requests.get')
        self.mock_get = self.patcher.start()
        self.mock_get.return_value.text = self.mock_response.getvalue()

    def tearDown(self):
        # Clean up the patcher
        self.patcher.stop()

    def test_dataframe_returned(self):
        # Ensure the function returns a DataFrame
        result = read_sellers_data()
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_exist(self):
        # Ensure the returned DataFrame has the expected columns
        expected_columns = ['mql_id', 'first_contact_date']
        result = read_sellers_data()
        self.assertCountEqual(result.columns.tolist()[:2], expected_columns)

    def test_columns_are_datetime(self):
        # Ensure the expected columns are of datetime type
        result = read_sellers_data()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['first_contact_date']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['won_date']))

class TestGetClosedDeals(unittest.TestCase):

    def setUp(self):
        # Set up a sample DataFrame for testing
        data = {
            'first_contact_date': pd.to_datetime(['2022-01-01', '2022-01-01', '2022-02-01', '2022-02-01']),
            'seller_id': [1, 2, 3, None]
        }
        self.df = pd.DataFrame(data)

    def test_dataframe_returned(self):
        # Ensure the function returns a DataFrame
        result = get_closed_deals(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_exist(self):
        # Ensure the returned DataFrame has the expected columns
        expected_columns = ['contact_year', 'contact_month', 'total_sales', 'closed_sales', 'percentage_closed_sales']
        result = get_closed_deals(self.df)
        self.assertCountEqual(result.columns.tolist(), expected_columns)

class TestGetClosedDealsPerOrigin(unittest.TestCase):

    def setUp(self):
        # Set up a sample DataFrame for testing
        data = {
            'origin': ['A', 'B', 'A', 'C', 'C'],
            'seller_id': [1, 2, None, 3, 4]
        }
        self.df = pd.DataFrame(data)

    def test_dataframe_returned(self):
        # Ensure the function returns a DataFrame
        result = get_closed_deals_per_origin(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_exist(self):
        # Ensure the returned DataFrame has the expected columns
        expected_columns = ['origin', 'total_deals', 'closed_deals', 'percentage_closed']
        result = get_closed_deals_per_origin(self.df)
        self.assertCountEqual(result.columns.tolist(), expected_columns)

class TestComputeAverageCloseTime(unittest.TestCase):

    def setUp(self):
        # Set up a sample DataFrame for testing
        data = {
            'first_contact_date': pd.to_datetime(['2022-01-01', '2022-01-01', '2022-02-01']),
            'won_date': pd.to_datetime(['2022-01-05', '2022-01-10', '2022-02-05']),
            'seller_id': [1, 2, 3]
        }
        self.df = pd.DataFrame(data)

    def test_float_returned(self):
        # Ensure the function returns a float
        result = compute_average_close_time(self.df)
        self.assertIsInstance(result, float)

    def test_average_calculation(self):
        # Ensure the average time difference is calculated correctly
        expected_average = 6.0
        result = compute_average_close_time(self.df)
        self.assertEqual(result, expected_average)

class TestComputeAvgDeclaredMonthlyRevenue(unittest.TestCase):

    def setUp(self):
        # Set up a sample DataFrame for testing
        data = {
            'declared_monthly_revenue': [1000, 2000, 3000, 4000],
            'seller_id': [1, 2, 3, 4]
        }
        self.df = pd.DataFrame(data)

    def test_float_returned(self):
        # Ensure the function returns a float
        result = compute_avg_declared_monthly_revenue(self.df)
        self.assertIsInstance(result, float)

    def test_average_calculation(self):
        # Ensure the average declared monthly revenue is calculated correctly
        expected_average = 2500.0
        result = compute_avg_declared_monthly_revenue(self.df)
        self.assertEqual(result, expected_average)

    def test_missing_values_handling(self):
        # Ensure the function handles missing values correctly
        df_with_na = pd.DataFrame({'declared_monthly_revenue': [1000, None, 2000],
                                   'seller_id': [1, 2, 3]})
        result = compute_avg_declared_monthly_revenue(df_with_na)
        self.assertEqual(result, 1500.0)

if __name__ == '__main__':
    unittest.main()
