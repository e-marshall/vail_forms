import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from organize_tables import RevExp, BuyBacks, EBITDA, ExecComp, ScrapeDict
from scrape import SECScraper

#make data 
target_url = 'https://data.sec.gov/submissions/CIK0000812011.json'
cutoff_date = "2014-01-01"
form_type_10k = '10-K'

scraper = SECScraper(target_url)
soup_dict_10k = ScrapeDict(scraper, target_url, cutoff_date, form_type_10k)

#mtn revenue + expenses
revenue_table_identifier = 'Mountain net revenue'
rev_exp_obj = RevExp(soup_dict_10k, revenue_table_identifier, 0)
revenue_df = rev_exp_obj.rev_df
revenue_df.to_csv('vail_revenue.csv')
expenses_df = rev_exp_obj.exp_df
expenses_df.to_csv('vail_expenses.csv')

#stock buybacks
buyback_table_identifier = "Repurchases of common stock"
buybacks_obj = BuyBacks(soup_dict_10k, buyback_table_identifier, 1)

buyback_df = buybacks_obj.buyback_df
buyback_df.to_csv('vail_buybacks.csv')

# executive compensation
target_url = 'https://data.sec.gov/submissions/CIK0000812011.json'
cutoff_date = "2014-01-01"
proxy_form_type = "DEF 14A"

discretionary_spending_identifier = 'Company-paid Lodging, Ski School Privileges and Discretionary Spending on Goods and Services'
#discretionary flag doesn't find any tables for some reason

base_salary_identifier1 = "Name and Principal Position"
base_salary_identifier = "Non-Equity Incentive Plan Compensation"

proxy_scraper = SECScraper(target_url)
proxy_soup_dict_b = proxy_scraper.run('2014-01-01', 'DEF 14A')
exec_compensation = ExecComp(proxy_soup_dict_b, base_salary_identifier,base_salary_identifier1)

full_ec = exec_compensation.full_df
full_ec.to_csv('vail_exec_compensation.csv')

# EBITDA
ebitda_table_identifier = 'Net income attributable to Vail Resorts, Inc.'

ebitda = EBITDA(soup_dict_10k, ebitda_table_identifier, 0,1, '2022-09-28')
ebitda_df = ebitda.full_df
ebitda_df.to_csv('vail_ebitda.csv')


def generate_revenue_plot():

    revenue_df = pd.read_csv('vail_revenue.csv')
    fig = px.line(
        revenue_df,
        x='year',
        y='amount',
        color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts Revenue (Total) and itemized",
        labels={"year": "Year", "amount": "Amount", "Item": "Item"}  # Updated label
    )
    return fig

def generate_stock_buyback_plot():

    buyback_df = pd.read_csv('vail_buybacks.csv')
    fig = px.line(
        buyback_df,
        x='Year',
        y='Value',
        #color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts Stock Buybacks",
        labels={"year": "Year", "amount": "Amount", "Item": "Item"}  # Updated label
    )
    return fig

def generate_ebitda_plot():

    ebitda_df = pd.read_csv('vail_ebitda.csv')
    fig = px.line(
        ebitda_df,
        x='Year',
        y='Amount',
        color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts EBITDA",
        labels={"year": "Year", "amount": "Amount", "Item": "Item"}  # Updated label
    )
    return fig

def generate_total_exec_compensation_plot():

    full_ec = pd.read_csv('vail_exec_compensation.csv')
    fig = px.line(
        full_ec.groupby(['Fiscal Year'])['Total'].sum().reset_index(),
        x='Fiscal Year',
        y='Total',
        #color='Name',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Executive Compensation",
        #labels={"year": "Year", "amount": "Amount", "Item": "Item"}  # Updated label
    )
    return fig

def generate_patrol_multiples_plot():

    exec_comp = pd.read_csv('vail_exec_compensation.csv')
    total_ec = full_ec.groupby(['Fiscal Year'])['Total'].sum().reset_index()

    patrol_base_wage = 21
    patrol_hours_guess = 880
    patrol_base_annual_comp = patrol_base_wage * patrol_hours_guess

    number_of_execs = {'2024': 6,
                   '2023': 7,
                   '2022': 6,
                   '2021': 6,
                   '2020': 5,
                   '2019': 5,
                   '2018': 5, 
                   '2017': 5,
                   '2016': 5,
                   '2015':6,
                   '2014':5}
    number_of_execs_df = pd.DataFrame(columns = ['Fiscal Year','Number of Executives'],
                                   data = number_of_execs.items())
    number_of_execs_df['Fiscal Year'] = number_of_execs_df['Fiscal Year'].astype(int)
    per_cap_exec_comp = pd.merge(total_ec, number_of_execs_df, on='Fiscal Year')
    per_cap_exec_comp['PerCapitaComp'] = per_cap_exec_comp['Total'] / per_cap_exec_comp['Number of Executives']
    per_cap_exec_comp['Multiples_of_patrol_annual_base_comp'] = per_cap_exec_comp['PerCapitaComp'] / patrol_base_annual_comp

    lynch_katz_comp = full_ec.loc[(full_ec['Name'].isin(['Kirsten A. Lynch','Robert A. Katz']))]
    lynch_katz_total = lynch_katz_comp.groupby(['Fiscal Year'])['Total'].sum().reset_index()
    lynch_katz_total['PerCap'] = lynch_katz_total['Total'] / 2
    lynch_katz_total['Multiples_of_patrol_annual_base_comp'] = lynch_katz_total['PerCap'] / patrol_base_annual_comp

    fig = px.line(
        per_cap_exec_comp,
        x='Fiscal Year',
        y='Multiples_of_patrol_annual_base_comp',
        markers=True,
        title="Per capita executive compensation as a multiple of patrol base compensation",
                )
    fig.data[0].name = 'All executives'
    fig.data[0].showlegend = True

    fig.add_trace(go.Scatter(
        x=lynch_katz_total['Fiscal Year'],
        y=lynch_katz_total['Multiples_of_patrol_annual_base_comp'],
        name='Lynch, Katz only',
        mode = 'lines+markers',
            ))

    # Change the y-axis range to start at 0
    fig.update_layout(
        yaxis=dict(
            range=[0, None]  # Start the y-axis at 0, and let the upper bound auto-adjust
                )
         )
    return fig

def main():
    st.title('Vail Resorts Earnings and Expenditures')
    st.subheader("How much money is Vail making? How much are they spending on stock buybacks and executive compensation? Food for thought after Vail refused to bargain in good faith with Park City Ski Patrollers for months?")
    st.markdown('This app looks at Vail Resorts earnings and expenditures based on SEC filings (Annual 10-K forms and annual meeting statements).')

    #st.subheader('Context')
    #st.markdown
    st.info('NOTE: this is in progress still, not finished yet')

    st.subheader('Vail Resorts Revenue (Total) and by category')
    st.markdown("Revenues have increased rapidly since 2020, with operations in the 'Lift' line item driving much of the revenue growth.")

    fig = generate_revenue_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Vail Resorts EBITDA')
    st.markdown('EBITDA = Earnings before interest, taxes, depreciation and amortization. \n [EBITDA](https://www.investopedia.com/terms/e/ebitda.asp) is a measure of profitability')
    st.markdown('Vail resorts has been pretty profitable recently ;)')
    fig = generate_ebitda_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Vail Resorts Stock Buybacks')
    st.markdown('How much is being spent on stock buybacks? Just a cool $725 million since 2020.')
    fig = generate_stock_buyback_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Vail Executive Compensation')
    st.markdown('This is total spent on compensation for all executives each year')
    st.markdown('There are 5 and 7 executives a year. In 2014 they made a combined 6.8 million. In 2022 it peaked at 24 million')
    fig = generate_total_exec_compensation_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Per capita executive compensation as a multiple of patrol base compensation')
    st.markdown('How many ski patrollers could you pay with the annual compensation of a single executive?')

    st.markdown('This is calculated by first finding the percapita executive compensation each year. Next, an estimate of annual wages earned by an entry level patroller making $21/hr')
    st.markdown('Vail executives make between 75-200x that of a patroller. Looking at the CEO and former CEO it is higher')
    fig = generate_patrol_multiples_plot()
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()



