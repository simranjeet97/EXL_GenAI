import streamlit as st
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai

#EXL logo
image = Image.open('exl.png')

# Load the CSV files
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

with st.sidebar:
	st.image(image, width = 100)
	st.title("Front End Load (FEL) Analysis")
    
actual_file = st.sidebar.file_uploader("Upload the Actuarial Data CSV File:", type=["csv"])
adjusted_file = st.sidebar.file_uploader("Upload the Finance Data CSV File:", type=["csv"])

# Defining Tabs
tab1, tab2 = st.tabs(['Data','Analysis'])

def get_expense_prop(actual_df):
    # Building Expense Proportions % AP
    Exp_AP = pd.DataFrame()
    Exp_AP['Commission %'] = actual_df['Commissions'] / actual_df['Annual_Premium'] * 100
    Exp_AP['PerPolicy %'] = actual_df['Per_Policy'] / actual_df['Annual_Premium'] * 100
    Exp_AP['PerSumAssured %'] = actual_df['Per_Sum_Assured'] / actual_df['Annual_Premium'] * 100
    Exp_AP['PerPremium %'] = actual_df['Per_Premium'] / actual_df['Annual_Premium'] * 100
    Exp_AP['Total %'] = actual_df['Total'] / actual_df['Annual_Premium'] * 100
    Exp_AP['LOB'] = actual_df['LOB']
    return Exp_AP

def get_diff(actual_df,adjusted_df):
    Exp_AP = get_expense_prop(actual_df)
    anum_pr = actual_df['Annual_Premium'].iloc[-1]
    # Differences between Actuarial and Finance Adjustment Numbers
    Diff_AF = pd.DataFrame()
    Diff_AF['Actuarial'] = actual_df['Annual_Premium'] / anum_pr * 100
    Diff_AF['Finance'] = adjusted_df['Annual_Premium'] / anum_pr * 100
    Diff_AF['Total Expense Ratio'] = Exp_AP['Total %'] #---------Against LOB
    Diff_AF['Change in Annual Premium (%)'] = Diff_AF['Finance'] - Diff_AF['Actuarial'] #-------Aginst LOB with cols
    Diff_AF['Impact_on_Ratio'] = round((Diff_AF['Change in Annual Premium (%)'] * Diff_AF['Total Expense Ratio'] * actual_df['Annual_Premium'].iloc[-1]) / 10000, 2)
    total_impact = Diff_AF['Impact_on_Ratio'].sum()
    Diff_AF.at[5, 'Impact_on_Ratio'] = total_impact
    Diff_AF['TotalExpRatio'] = adjusted_df['Total'] / adjusted_df['Annual_Premium'] * 100
    Diff_AF['LOB'] = actual_df['LOB']
    return Diff_AF


with tab1:
    try:
        actual_df = load_data(actual_file)
        adjusted_df = load_data(adjusted_file)
    except Exception as e:
        st.write("")

    if actual_file and adjusted_file:
        # Display the results
        st.subheader("Actuarial")
        st.dataframe(actual_df)
        st.markdown(f"*All Numbers is in Billions")

        st.subheader("Finance Adjustment")
        st.dataframe(adjusted_df)
        st.markdown(f"*All Numbers is in Billions. ")

        new_total = st.number_input("Enter New Finance Total:", min_value=0.01, value=adjusted_df.loc[adjusted_df['LOB'] == 'Total', 'Annual_Premium'].values[0])

        # Calculate the proportion of each LOB's 'Annual_Premium' in the 'Total' and express it as a percentage
        if 'Total' in adjusted_df['LOB'].values:
            total_annual_premium = adjusted_df[adjusted_df['LOB'] == 'Total']['Annual_Premium'].values[0]
            adjusted_df['Proportion (%)'] = (adjusted_df['Annual_Premium'] / total_annual_premium) * 100

        # Update the 'Annual_Premium' values for all LOBs based on the proportion
        adjusted_df['Annual_Premium'] = adjusted_df.apply(lambda row: new_total * row['Proportion (%)']/100 if row['LOB'] != 'Total' else new_total, axis=1)


        st.subheader("Updated Finance Adjustment")
        # Display the updated DataFrame
        st.dataframe(adjusted_df)

    else:
        st.write("Please Upload the Files.")

    

with tab2:
    try:
        Diff_AF = get_diff(actual_df,adjusted_df)
        BusinessMixImpact = round(Diff_AF['Impact_on_Ratio'].iloc[-1], 2)
        SalesImpact = round((adjusted_df['Annual_Premium'].iloc[-1] - actual_df['Annual_Premium'].iloc[-1]) * Diff_AF['TotalExpRatio'].iloc[-1] / 100, 2)    
        OverallImpact = round((BusinessMixImpact + SalesImpact),2)
    except:
        st.write("Please Upload the Files.")
        
    if st.button("Get Analysis"):
        Diff_AF = Diff_AF[['LOB','Total Expense Ratio','Change in Annual Premium (%)']]
        Show_Dataframe = Diff_AF.copy()
        Show_Dataframe['Change in Annual Premium (%)'] =  Show_Dataframe['Change in Annual Premium (%)'].apply(lambda x: str(int(x))+" %")
        Show_Dataframe['Total Expense Ratio'] =  Show_Dataframe['Total Expense Ratio'].apply(lambda x: str(int(x))+" %")

        st.markdown("#### LOB Wise Total Expense Ratio and Change in Annual Premium (%):")
        col1, col2 = st.columns(2)    
        col1.dataframe(Show_Dataframe[['LOB','Total Expense Ratio']])
        col2.dataframe(Show_Dataframe[['LOB','Change in Annual Premium (%)']])

        color_scale1 = 'Viridis'  
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Expense Ratio (%)", "Change in Annual Premium (%)"))
        fig.add_trace(go.Bar(x=Diff_AF['LOB'], y=Diff_AF['Total Expense Ratio'], marker=dict(color=Diff_AF['Total Expense Ratio'], colorscale=color_scale1)), row=1, col=1)
        fig.add_trace(go.Bar(x=Diff_AF['LOB'], y=Diff_AF['Change in Annual Premium (%)'], marker=dict(color=Diff_AF['Change in Annual Premium (%)'], colorscale=color_scale1)), row=1, col=2)
        fig.update_layout(title="Total Expense Ratio (%) and Change in Annual Premium (%)", showlegend=False)
        st.plotly_chart(fig)

        st.markdown("#### GenAI Financial Analysis: ")
        st.write(f"Business Mix Impact: ${BusinessMixImpact}","Bn" )
        st.write(f"Sales Impact: ${SalesImpact}","Bn")
        st.write(f"Overall Impact: ${OverallImpact}","Bn")

        actuarial_value = actual_df.loc[actual_df.LOB == 'Total','Annual_Premium'].values[0]
        adjusted_value = adjusted_df.loc[adjusted_df.LOB == 'Total','Annual_Premium'].values[0]
        st.markdown(f"- Total Sales for Actuarial is: {actuarial_value}")
        st.markdown(f"- Total Sales for Finance is: {adjusted_value}")

        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15, 5))
        # ax1.set_title("Total Expense Ratio for Each LOB")
        # ax1.bar(Diff_AF['LOB'], Diff_AF['Total Expense Ratio'])
        # ax2.set_title("Change in Annual Premium (%) for Each LOB")
        # ax2.bar(Diff_AF['LOB'],Diff_AF['Change in Annual Premium (%)'])
        # st.pyplot(fig)

        # Create a Seaborn barplot for the continuous data
        # sns.set(style="whitegrid")
        # plt.figure(figsize=(10, 6))

        # # Barplot
        # sns.barplot(x='LOB', y='Total Expense Ratio', data=Diff_AF, color='lightblue')

        # # Line plot on the same axes
        # sns.lineplot(x='LOB', y='Change in Annual Premium (%)', data=Diff_AF, marker='o', color='red')

        # plt.title('Change in Annual Premium with Total Expense Ratio for Each LOB')
        # plt.xlabel('LOB')
        # plt.ylabel('Total Expense Ratio')

        # # ----

        # # Create a Seaborn barplot for the continuous data
        # sns.set(style="whitegrid")
        # plt.figure(figsize=(10, 6))

        # # Line plot on the same axes
        # sns.lineplot(x='LOB', y='Change in Annual Premium (%)', data=Diff_AF, marker='o', color='red')

        # plt.title('Change in Annual Premium with Total Expense Ratio for Each LOB')
        # plt.xlabel('LOB')
        # plt.ylabel('Total Expense Ratio')

        # Convert the DataFrame to a text format
        data = {
            'Business Mix Impact': [BusinessMixImpact],
            'Sales Impact': [SalesImpact],
            'Overall Impact': [OverallImpact]
        }
        EIR_data = pd.DataFrame(data)
        #eir_text = EIR_data.to_csv(index=False)
        #st.dataframe(EIR_data)


        # Define the prompt
        prompt =  f"""
            You are a chief finance officer in an insurance company. Provide an analysis on the
            expense impact report (in billions): {EIR_data},
            taking reasoning and cause from the finance data: {Diff_AF}
            Talk about business shift, and sales.
            Make the analysis very concise in 2-4 sentences. The impact we are talking about is business expenses 
            due to financial adjustments. Say in confident tone, which can be put in MIS report
        """

        #st.write(prompt)

        def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.9): 
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                api_key=st.secrets.api_key
            )
            return response.choices[0].message["content"]

        # # def get_analysis(api_key,prompt): 
        # #     # Use the OpenAI API to generate differences
        # #     response = openai.Completion.create(
        # #         engine="text-davinci-002",  # You can use a different engine as needed
        # #         prompt=prompt,
        # #         max_tokens=300,  # Adjust as needed
        # #         api_key=api_key,
        # #         temperature=0.7
        # #     )

        # #     # Extract the differences from the API response
        # #     analysis = response.choices[0].text
        # #     return analysis

        # # analysis = get_analysis(Diff_AF,eir_text,api_key)
        completition = get_completion(prompt)
        # # st.write(analysis)
        # # st.markdown("#### GenAI Financial Analysis:")
        st.write(completition)
        with open('dump.txt','a') as file:
            file.write(completition)
            file.write("\n\n")
        # # print(analysis)            