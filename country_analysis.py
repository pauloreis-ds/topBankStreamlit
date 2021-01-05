import pandas as pd


def _format_percentage(value1, value2, value3):
    value1 = str(round(value1 * 100, 2)) + "%"
    value2 = str(round(value2 * 100, 2)) + "%"
    value3 = str(round(value3 * 100, 2)) + "%"
    return value1, value2, value3


def get_churn_rate(country_1, country_2, country_3):
    rate_1 = (country_1.groupby(['geography']).exited.mean())[0]
    rate_2 = (country_2.groupby(['geography']).exited.mean())[0]
    rate_3 = (country_3.groupby(['geography']).exited.mean())[0]

    country_1_churn_rate, country_2_churn_rate, country_3_churn_rate = _format_percentage(rate_1, rate_2, rate_3)

    return pd.DataFrame({country_1['geography'].unique()[0]: [country_1_churn_rate],
                         country_2['geography'].unique()[0]: [country_2_churn_rate],
                         country_3['geography'].unique()[0]: [country_3_churn_rate]},
                        index=['Churn Rate'])


def get_active_members_rate(country_1, country_2, country_3):
    rate_1 = (country_1.groupby('geography').is_active_member.mean())[0]
    rate_2 = (country_2.groupby('geography').is_active_member.mean())[0]
    rate_3 = (country_3.groupby('geography').is_active_member.mean())[0]

    country_1_active_rate, country_2_active_rate, country_3_active_rate = _format_percentage(rate_1, rate_2, rate_3)

    return pd.DataFrame({country_1['geography'].unique()[0]: [country_1_active_rate],
                         country_2['geography'].unique()[0]: [country_2_active_rate],
                         country_3['geography'].unique()[0]: [country_3_active_rate]},
                        index=['Active Members'])


def get_credit_card_rate(country_1, country_2, country_3):
    rate_1 = (country_1.groupby(['geography']).has_cr_card.mean())[0]
    rate_2 = (country_2.groupby(['geography']).has_cr_card.mean())[0]
    rate_3 = (country_3.groupby(['geography']).has_cr_card.mean())[0]

    country_1_credit_card_rate, country_2_credit_card_rate, country_3_credit_card_rate = _format_percentage(rate_1,
                                                                                                            rate_2,
                                                                                                            rate_3)

    return pd.DataFrame({country_1['geography'].unique()[0]: [country_1_credit_card_rate],
                         country_2['geography'].unique()[0]: [country_2_credit_card_rate],
                         country_3['geography'].unique()[0]: [country_3_credit_card_rate]},
                        index=['Has Credit Card'])


def get_churn_chart(country):
    return pd.DataFrame({'Total Customers': country.groupby('age').exited.count(),
                         'Churned Customers': country.groupby('age').exited.sum()},
                        index=country.groupby('age').exited.sum().index)


def get_revenue_age_chart(country):
    annual_revenue_per_age = country.groupby('age').sum()['annual_revenue']

    return pd.DataFrame({'Sum of Revenue': annual_revenue_per_age,
                         '': annual_revenue_per_age.index},
                        index=annual_revenue_per_age.index)


def get_number_of_products_chart(country):
    number_of_products_count = country.groupby('num_of_products').count()['gender']

    return pd.DataFrame({'Number of Products': number_of_products_count},
                        index=number_of_products_count.index)


def get_churn_tenure(country):
    churn_by_tenure = country.groupby('tenure').exited.mean()

    return pd.DataFrame({'Churn buy tenure': churn_by_tenure},
                        index=churn_by_tenure.index)