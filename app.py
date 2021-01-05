import pandas as pd
import streamlit as st
from model import Model
from country_analysis import *


def load_process_data():
    raw_data = pd.read_csv("./data/churn_data.csv")

    TWELVE_MONTHS = 12
    mean_salary = raw_data['estimated_salary'].mean()
    raw_data['annual_revenue'] = [(salary / TWELVE_MONTHS) * 0.2 if salary > mean_salary else (salary / TWELVE_MONTHS) *
                                                                                              0.15 for salary in
                                  raw_data['estimated_salary']]

    raw_data['revenue_per_product'] = raw_data['annual_revenue'] / raw_data['num_of_products']

    raw_data['total_revenue'] = [tenure * annual_earning if tenure > 0 else annual_earning
                                 for tenure, annual_earning in zip(raw_data['tenure'], raw_data['annual_revenue'])]
    data = raw_data.drop(columns='row_number')
    return data


def convert_to_numeric(value):
    value = value.replace(',', '.')
    try:
        value = float(value)
        if (value > 1) and (value <= 100):
            return value / 100
        elif value > 100:
            st.warning("That doesn't make sense '-'")
        elif value < 0:
            return 0

        return value
    except:
        st.error("Come on! This is Not a Valid Value  (-.-)")


if __name__ == "__main__":
    img = 'https://raw.githubusercontent.com/pauloreis-ds/Paulo-Reis-Data-Science/master/Paulo%20Reis/PRojects.png'
    st.set_page_config(page_title='TopBank Churn Project', page_icon=img)
    data_frame = load_process_data()
    model = Model()

    menu_option = st.sidebar.selectbox('Options', ['Model Usage', 'Country Initial Analysis'])

    if menu_option == 'Model Usage':
        st.header('Model Usage')
        st.write('''If TopBank, knowing that a customer is going to leave the company, 
                     can prevent that from happening...''')
        st.write('''__"We want to be able to figure out accurately  which customer is going to churn.
                  Therefore, recall is the metric that best fits this problem."__''')
        st.write("_From those who were churned customers, how many did the model detect?_ 88%")

        st.write('''_"Let's say topBank can't prevent customers who are about to leave from leaving... 
                    how can we calculate the revenue for this case? We'll use the probability of churn. 
                    If the probability of a customer leave the company is too high, then we'll consider 
                    it as a churned customer already. However, actually, for us, it will be considered as
                    a not churned customer, so the model will not detect it, leading the company to not 
                    knowing there will be churn from that person. And if TopBank doesn't know, it can't 
                    change that customer behavior"_''')

        sample = data_frame
        predictions = model.predict(sample)
        predictions_probabilities = model.predict_probability(sample)

        # Predicted as churn or not + probability of turning into a churned customer
        churn_predictions = pd.Series(predictions, name='churn_prediction')
        churn_probabilities = pd.Series(predictions_probabilities, name='churn_probability')

        model_results = pd.concat([churn_predictions, churn_probabilities], axis=1)
        threshold_value = st.text_input('Threshold Value. We can prevent them from leaving if their churn probability '
                                        'is is up to...', 1.0)
        threshold_value = convert_to_numeric(threshold_value)
        st.info(f'''Every probability greater than {round(threshold_value * 100, 2)}% is going to be considered 
                    as a not (detected) churn, meaning the company will fail in preventing that customer from 
                    leaving.''')

        # Changing the predictions based on probability of churn and threshold_value
        reformatted_predictions = model_results['churn_probability'].apply(lambda x: 1 if (x > 0.5) and
                                                                                          (x <= threshold_value) else 0)
        reformatted_predictions = pd.Series(reformatted_predictions, name='adapted_prediction')

        adapted_model_results = pd.concat([model_results, reformatted_predictions], axis=1)

        st.write("Model's results:")

        if st.checkbox("Static Sample"):
            head_tail = pd.concat([adapted_model_results.head(), adapted_model_results.tail()])
            st.dataframe(head_tail)
        else:
            ten_samples = pd.concat([adapted_model_results.sample(5), adapted_model_results.sample(5)])
            st.dataframe(ten_samples)

        st.write('Quantity of Detected Churned Customers Predictions')
        st.write(adapted_model_results[['churn_prediction', 'adapted_prediction']].sum().astype(str))

        # These are the 88% we detect. How many is the 100%?
        sum_of_predicted_exit = adapted_model_results.sum()[0]
        # math - rule of three
        actual_exited = sum_of_predicted_exit * 1 / 0.88
        # Detected based on the new threshold, so we can calculate the new recall
        adapted_churned_detections = adapted_model_results.sum()[2]

        # Recall - From those who were churned customers, how many did the model detect?
        new_recall = adapted_churned_detections / actual_exited
        if new_recall >= 0.8:
            st.success(f"Actual Recall {new_recall.round(2) * 100}%")
        elif (new_recall < 0.8) and (new_recall > 0.7):
            st.info(f"Actual Recall {new_recall.round(2) * 100}%")
        else:
            st.error(f"Actual Recall {new_recall.round(2) * 100}%")

        expected_revenue_per_customer = sample.annual_revenue.mean()
        churned_customers = 2000  # 20% of 10,000 customers
        expected_revenue_return = (churned_customers * expected_revenue_per_customer) * new_recall
        st.write("Expected revenue return {:0,.2f} EUR".format(expected_revenue_return))
        st.write('More Details at the End of [topBankModel notebook](https://nbviewer.jupyter.org/github/pauloreis-ds/'
                 'Projetos/blob/master/classification-churn/notebooks/topBankModel.ipynb)')

    else:
        if st.checkbox("Display CEO's Report"):
            st.write('''_"At the end of your consultancy, you need to deliver to the TopBank CEO a model in production,
                        which will receive a customer base via API and return that same base scored, that is, one more
                         column with the probability of each customer entering into churn."_''')
            report_data = data_frame.sample(10)
            prediction_probability = model.predict_probability(report_data)

            probabilities = pd.Series(prediction_probability * 100, name="churn_probability", index=report_data.index)
            probabilities = probabilities.round(2).apply(lambda x: str(x) + '%')

            ceo_report = pd.concat([report_data, probabilities], axis=1)

            to_drop = ['exited', 'revenue_per_product', 'annual_revenue', 'total_revenue']
            st.dataframe(ceo_report.drop(columns=to_drop).reset_index(drop=True))

        st.header('Country Analysis - An Initial Approach')
        df = data_frame.drop(columns=['customer_id', 'surname', 'revenue_per_product', 'total_revenue'])
        spain = df.query("geography == 'Spain'")
        france = df.query("geography == 'France'")
        germany = df.query("geography == 'Germany'")

        st.subheader('Basic Percentages')
        col_1, col_2, col_3 = st.beta_columns(3)
        col_1.write(get_churn_rate(france, germany, spain).T)
        col_2.write(get_active_members_rate(france, germany, spain).T)
        col_3.write(get_credit_card_rate(france, germany, spain).T)
        st.write('\n\n')

        st.subheader("Customers Count by Age")
        country = st.selectbox(' ', ['France', 'Germany', 'Spain'])
        if country == 'France':
            chart_data = get_churn_chart(france)
            st.area_chart(chart_data)
            st.write("Stats")
            st.dataframe(france.describe().T.round(2).astype(str))
        elif country == 'Germany':
            chart_data = get_churn_chart(germany)
            st.area_chart(chart_data)
            st.write("Stats")
            st.dataframe(germany.describe().T.round(2).astype(str))
        elif country == 'Spain':
            chart_data = get_churn_chart(spain)
            st.area_chart(chart_data)
            st.write("Stats")
            st.dataframe(spain.describe().T.round(2).astype(str))

        st.subheader("Number of Products Sales")
        if st.checkbox("Display "):
            st.write("France")
            chart_data = get_number_of_products_chart(france)
            st.bar_chart(chart_data)
            st.write("Germany")
            chart_data = get_number_of_products_chart(germany)
            st.bar_chart(chart_data)
            st.write("Spain")
            chart_data = get_number_of_products_chart(spain)
            st.bar_chart(chart_data)

        st.subheader("Churn Rate by Tenure")
        if st.checkbox("Display  "):
            chart_data = get_churn_tenure(france)
            st.line_chart(chart_data)
            st.write("Germany")
            chart_data = get_churn_tenure(germany)
            st.line_chart(chart_data)
            st.write("Spain")
            chart_data = get_churn_tenure(spain)
            st.line_chart(chart_data)

        st.subheader("Sum of Annual Revenue by Age")
        if st.checkbox("Display"):
            st.write("France")
            chart_data = get_revenue_age_chart(france)
            st.bar_chart(chart_data)
            st.write("Germany")
            chart_data = get_revenue_age_chart(germany)
            st.bar_chart(chart_data)
            st.write("Spain")
            chart_data = get_revenue_age_chart(spain)
            st.bar_chart(chart_data)
