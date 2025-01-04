import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from organize_tables import RevExp, BuyBacks, EBITDA, ExecComp, ScrapeDict
from scrape import SECScraper




def generate_revenue_plot():

    revenue_df = pd.read_csv('vail_revenue.csv')
    fig = px.line(
        revenue_df,
        x='year',
        y='amount',
        color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts Revenue (Total) and itemized",
        labels={"year": "Year", "amount": "Amount ($)", "Item": "Item"},  # Updated label
        color_discrete_sequence=px.colors.sequential.Blues

    )
    return fig

def generate_expenses_plot():
    expenses_df = pd.read_csv('vail_expenses.csv')
    fig = px.line(
        expenses_df,
        x='year',
        y='amount',
        color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts Expenses (Total) and itemized",
        labels={"year": "Year", "amount": "Amount ($)", "Item": "Item"}, # Updated label
        color_discrete_sequence=px.colors.sequential.Reds
    )
    return fig

def generate_combined_plot():

    revenue_fig = generate_revenue_plot()
    expenses_fig = generate_expenses_plot()

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Revenue", "Expenses"))
    for trace in revenue_fig['data']:
        trace['legendgroup'] = 'Revenue'
        trace['showlegend'] = True
        fig.add_trace(trace, row=1, col=1)

    for trace in expenses_fig['data']:
        trace['legendgroup'] = 'Expenses'
        trace['showlegend'] = True
        fig.add_trace(trace, row=1, col=2)
    
    fig.update_layout(
        title_text="Revenue and Expenses",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1.,
            xanchor="center",
            x=0.5,
            traceorder="grouped",
            title_text="",
            font=dict(size=10),
            itemwidth=30,
            tracegroupgap=20
        )
    )

    # Update legend items to be in two columns per group
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-2.5,
            xanchor="center",
            x=0.5,
            traceorder="grouped",
            title_text="",
            font=dict(size=10),
            itemwidth=30,
            tracegroupgap=20
        )
    )
    return fig
def generate_stock_buyback_plot():

    buyback_df = pd.read_csv('vail_buybacks.csv')
    fig = px.line(
        buyback_df.rename({'Value': 'Amount ($)'}, axis=1),
        x='Year',
        y='Amount ($)',
        #color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts Stock Buybacks",
        labels={"year": "Year", "amount": "Amount ($)", "Item": "Item"},  # Updated label
    )
    return fig

def generate_ebitda_plot():

    ebitda_df = pd.read_csv('vail_ebitda.csv')
    fig = px.line(
        ebitda_df.rename({'Amount':'Amount ($)'}, axis=1),
        x='Year',
        y='Amount ($)',
        color='Item',  # Updated to match your DataFrame column name
        markers=True,
        title="Vail Resorts EBITDA",
        labels={"year": "Year", "amount": "Amount ($)", "Item": "Item"}  # Updated label
    )
    update_fig = fig.update_layout(showlegend=False)
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
        labels={"Fiscal Year": "Year", "Total": "Amount ($)", "Item": "Item"}  # Updated label
    )
    return fig

def generate_patrol_base_bubble_plot():

    exec_comp = pd.read_csv('vail_exec_compensation.csv')

    full_ec = exec_comp
    patrol_base_wage = 21
    req_patrol_base_wage = 23
    annual_hours = 40*50
    bubble_df = pd.DataFrame({'Annual_starting_patrol_comp': patrol_base_wage * annual_hours,
                            'Mean CEO compensation': np.round(full_ec.loc[(full_ec['Fiscal Year'].isin([2020,2021,2022,2023,2024])) & (full_ec['Name'] == 'Kirsten A. Lynch')]['Total'].mean()),
                            'Requested_annual_starting_patrol_comp': req_patrol_base_wage * annual_hours}, index=[0])

    fig = go.Figure()

    # Add the first bubble trace
    fig.add_trace(go.Scatter(
        x=[25],  # Arbitrary x position
        y=[10],  # y-axis set to zero
        mode='markers',
        marker=dict(size=bubble_df['Annual_starting_patrol_comp']/8000, color='Blue'),  # Adjust size for better visualization
        name='Starting_patrol_compensation (Assuming full year)',
        text=[f"Starting_patrol_compensation (Assuming full year): ${bubble_df['Annual_starting_patrol_comp'][0]:,}"],  # Text to display on hover with formatted number
        hoverinfo='text'  # Display only the text on hover
        
    ))
    fig.add_trace(go.Scatter(
        x=[125],  # Arbitrary x position
        y=[10],  # y-axis set to zero
        mode='markers',
        marker=dict(size=bubble_df['Requested_annual_starting_patrol_comp']/8000, color='Green'),  # Adjust size for better visualization
        name='Requested starting_patrol compensation (Assuming full year)',
        text=[f"Requested starting_patrol_compensation (Assuming full year): ${bubble_df['Requested_annual_starting_patrol_comp'][0]:,}"],  # Text to display on hover with formatted number
        hoverinfo='text'  # Display only the text on hover
        
    ))

    # Add the second bubble trace
    fig.add_trace(go.Scatter(
        x=[500],  # Shifted x position
        y=[-10],  # y-axis set to zero
        mode='markers',
        marker=dict(size=bubble_df['Mean CEO compensation']/8000, color='Red'),  # Adjust size for better visualization
        name='Average annual executive compensation (2020-2024)',
        text=[f"'Average annual CEO compensation \n (2020-2024)': ${bubble_df['Mean CEO compensation'][0]:,}"],  # Text to display on hover with formatted number
        hoverinfo='text'  # Display only the text on hover
        
    ))

    # Update layout to hide axes and grid
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1000]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',  # Remove plot background color
        paper_bgcolor='rgba(0,0,0,0)' ,
        height=900, width = 1200,
        margin=dict(l=10, r=10, t=30, b=10),
        autosize=True,
        #responsive=True,
        legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Align the bottom of the legend to the plot
        y=1.02,            # Place it just above the plot (1.0 is the top)
        xanchor="center",  # Center the legend horizontally
        x=0.5              # Position it at the center of the plot
    ),
    )
    return fig

    




def generate_patrol_multiples_plot():

    exec_comp = pd.read_csv('vail_exec_compensation.csv')

    full_ec = exec_comp
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
            range=[0, None],  # Start the y-axis at 0, and let the upper bound auto-adjust
            title= 'Equivalent # of base ski patrol annual compensation amounts'
                )
         )
    return fig

def main():
    st.title('Vail Resorts Earnings and Expenditures')
    st.divider()
    st.header("How much money is Vail making?")
    st.subheader("Food for thought amidst the ongoing work stoppage of Park City Professional Ski Patrol Association over a failure to reach a new contract with Vail Resorts.")
    st.markdown('This app looks at Vail Resorts earnings and expenditures based on data disclosed in 10-K and annual meeting statements. All data used is publicly available through the [SEC EDGAR database](https://www.sec.gov/search-filings).')

    st.divider()

    #st.header("How much are they spending on stock buybacks and executive compensation?")
    st.subheader("A little bit of background")
    st.markdown("On December 27, 2024, the [Park City Professional Ski Patrol Association went on strike](https://www.sltrib.com/news/2024/12/27/park-city-mountain-ski-patrollers/). They have been attempting to negotiate a new contract with Vail Resorts since their previous contract expired in April 2024. During this time, [multiple unfair labor practice complaints](https://www.kpcw.org/ski-resorts/2024-12-19/park-city-mountain-ski-patrol-union-files-unfair-labor-practice-complaints-against-vail) have been filed against the company with the National Labor Relations Board (NLRB), and the [company has failed to provide counter offers to the Union's proposed wages and benefits package, offering only an effective 0.5% increase](https://www.parkrecord.com/2024/12/22/ski-patrol-union-contract-negotiations-reach-agreement-on-24th-of-27-items-but-not-the-main-one-pay/).")
    st.markdown("After seeing some numbers shared in news articles and on social media, I was curious to learn more about how much money the publicly-traded company, Vail Resorts (which owns Park City Mountain) was making and other aspects of their finances. I wanted to learn a bit more about web scraping and some other Python tools so I decided to see if I could make an app based on financial data extracted from Vail's SEC filings.")
    st.divider()
    #st.subheader('Context')
    #st.markdown
    st.info('NOTE: this is in progress still, not finished yet')
    #st.info('')

    st.subheader('Revenue and expenses: total and by category')
    st.markdown("Revenues have increased rapidly since 2021, with operations in the 'Lift' line item driving much of the revenue growth.")
    st.markdown('Vail Resorts has increased its investment in labor and labor-related expenses in recent years.')

    fig = generate_combined_plot()
    st.plotly_chart(fig, use_container_width=True)

    #st.subheader('Expenses (total) and by category')
    #st.markdown('Vail Resorts has increased its investment in labor and labor-related expenses in recent years')
    #fig = generate_expenses_plot()
    #st.plotly_chart(fig, use_container_width=True)

    st.subheader('Vail Resorts EBITDA')
    st.markdown('Earnings before interest, taxes, depreciation and amortization (EBITDA). \n [EBITDA](https://www.investopedia.com/terms/e/ebitda.asp) is a measure of profitability')
    #st.markdown('Vail Resorts has been pretty profitable recently...')
    fig = generate_ebitda_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Vail Resorts Stock Buybacks')
    st.markdown('How much is being spent on stock buybacks? Just a cool $725 million since 2020.')
    st.markdown('Here are some articles that discuss the role of stock buybacks from a range of perspectives: [More Perfect Union](https://perfectunion.us/stock-buybacks-good-for-warren-buffett-bad-for-working-people/), [Harvard Law Today](https://hls.harvard.edu/today/whats-the-deal-with-stock-buybacks/).')
    fig = generate_stock_buyback_plot()
    st.plotly_chart(fig, use_container_width=True)
    #https://perfectunion.us/stock-buybacks-good-for-warren-buffett-bad-for-working-people/

    st.subheader('Vail Executive Compensation')
    st.markdown('This is the total amount spent on compensation for all executives each year.')
    st.markdown('Depending on the year, there are between 5 and 7 executives in this category. In 2014, they made a combined 6.8 million. In 2022, total executive compensation peaked at 24 million.')
    fig = generate_total_exec_compensation_plot()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('How does CEO compensation compare to ski patrol compensation?')
    st.markdown('To make this plot, I used to starting hourly wage for a ski patroller under the previous contract and the starting wage that they are requesting in the current negotiations. I calculated the annual compensation someone would make at that wage working 50 weeks a year in order to compare it to the annual CEO compensation.')
    st.markdown('Most ski patrollers work seasonally, this is not meant to be an accurate representation of their annual compensation from Vail Resorts; it is intended to compare the previous and requested wages to CEO compensation.')
    st.info("If you're on mobile, this figure might not render well. ")
    fig = generate_patrol_base_bubble_plot()
    st.plotly_chart(fig, use_container_width=True)

    
    st.divider()
    st.markdown("Disclaimer: I'm not a financial analyst etc. etc., this was done on my personal time.")
    st.markdown('All code used in this app is available [here](https://github.com/e-marshall/vail_forms)')



if __name__ == "__main__":
    main()



