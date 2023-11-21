import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go 

# Page layout
## Page expands to full width
st.set_page_config(
    page_title='CPI Liverpool demo',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)

st.title("CPI x Liverpool digital innovation hub proposal")


import streamlit as st

tab1, tab2 = st.tabs(["Heart Failure stats", "🫀 ECG demo"])


# ------------------------------------------------------------------------------

with tab2: 
    def display_patient_ecg(patient_id, abnormality_probabilities, abnormality_threshold=0.5):
        st.write(f'Patient: {patient_id}')
        data = pd.read_csv(f'{patient_id}.csv')
        
        # Classify segments based on the abnormality threshold
        classified_segments = np.argmax(abnormality_probabilities, axis=1)
        
        # Calculate the proportion of segments for each class
        class_counts = np.bincount(classified_segments, minlength=3)
        total_segments = len(abnormality_probabilities)
        proportions = class_counts / total_segments
        
        st.write("Proportion of ECG segments:")
        st.write(f"Normal: {proportions[0]:.2%}")
        st.write(f"Supraventricular: {proportions[1]:.2%}")
        st.write(f"Ventricular: {proportions[2]:.2%}")

        fig, axes = plt.subplots(3, 2, figsize=(5, 5))

        for i in range(3):
            for j in range(2):
                start_second = (i * 2 + j) * 5
                start_sample = int(start_second * 1000)
                end_sample = start_sample + 5000

                data_subset = data[start_sample:end_sample]

                ax = sns.lineplot(data=data_subset['MLII'], ax=axes[i, j], color='red')
                ax.set(ylabel='Voltage in mV', xlabel='time in seconds')
                ax.set_title(f"Segment from {start_second}-{start_second + 5} seconds")

                # Convert x-axis ticks from milliseconds to seconds
                x_ticks = ax.get_xticks()
                ax.set_xticklabels([f'{int(tick/1000)}' for tick in x_ticks])

        plt.tight_layout()
        st.pyplot(fig)



        st.markdown('#') 
        st.markdown('#') 
        st.markdown('#') 

        # Display the table of predicted probabilities
        table_data = pd.DataFrame({
            'Segment': [f'Segment {i+1}' for i in range(len(abnormality_probabilities))],
            'Normal': abnormality_probabilities[:, 0],
            'Supraventricular': abnormality_probabilities[:, 1],
            'Ventricular': abnormality_probabilities[:, 2],
            'Classification': classified_segments
        })

        # Display segments that were classified as NOT Normal
        st.header("Segments classified as abnormal:")
        not_normal_segments = table_data[table_data['Classification'] != 0]

        # Create a dropdown with segments classified as NOT Normal
        selected_not_normal_segment = st.selectbox("Select Abnormal segment:", not_normal_segments['Segment'])

        # Plot the selected NOT Normal segment
        if st.button("Plot Selected Abnormal Segment"):
            selected_not_normal_segment_index = int(selected_not_normal_segment.split()[1]) - 1
            plot_segment(data, selected_not_normal_segment_index, abnormality_probabilities[selected_not_normal_segment_index])

    def plot_segment(data, segment_number, abnormality_probabilities):
        start_second = segment_number * 3.6
        start_sample = int(start_second * 1000)
        end_sample = start_sample + 5000

        data_subset = data[start_sample:end_sample]

        plt.figure(figsize=(4, 3))
        plt.plot(data_subset['MLII'], color='red')
        plt.xlabel('Time in seconds')
        plt.ylabel('Voltage in mV')
        plt.title(f"Abnormal Segment {segment_number + 1}")

        # Convert x-axis ticks from milliseconds to seconds
        x_ticks = plt.xticks()[0]
        plt.xticks(x_ticks, [f'{int(tick / 1000)}' for tick in x_ticks])
        plt.tight_layout()
        st.pyplot(plt)

    st.header("View patient ECGs")

    # option = st.selectbox("Select patient:", ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", "113", "114", "115"])
    option = st.selectbox("Select patient:", ["100", "101", "105", "106", "108", "109", "114", "115", "116", "118"])
    abnormality_probabilities = np.load(f'{option}.npy')

    # Set an abnormality threshold (adjust as needed)
    abnormality_threshold = 0.5

    display_patient_ecg(option, abnormality_probabilities, abnormality_threshold)


#------------------------------------------------------------------------------

# Heart failure statistics and demo interactive plots 

with tab1: 

    # Using the population statistics of Liverpool: 
    def HF_thirty_day_readmissions(year): 
        np.random.seed(year)
        total_HF_pop = 5900+(year-2022)*100
        readmitted = (round(total_HF_pop/100)*18)
        return readmitted

    # Function to generate a baseline vector for a given year
    def generate_baseline_vector(year):
        # Set seed for reproducibility
        np.random.seed(year)
        
        # Generate a random vector of integers
        vector_size = HF_thirty_day_readmissions(year)
        random_vector = np.random.randint(1, 12, size=vector_size)

        # Calculate the current mean
        current_mean = np.mean(random_vector)

        # Adjust the vector to have the desired mean
        random_vector = random_vector + (6 - current_mean)

        # Calculate the new mean after adjusting
        new_mean = np.mean(random_vector)

        # Skew the distribution while keeping the mean constant
        skew_factor = 0.5
        skewed_vector = np.round(random_vector + skew_factor * (random_vector - new_mean)).astype(int)

        # Ensure non-negative values
        baseline_vector = np.maximum(0, skewed_vector)
        
    #     indices_to_remove = np.random.choice(vector_size, size=round(vector_size/4), replace=False)
    #     intervention_vector = np.delete(baseline_vector, indices_to_remove)

        return baseline_vector




    # Create an empty DataFrame
    baseline_df = pd.DataFrame()

    # Initialize a list to store total bed days for each year
    total_bed_days_list = []
    total_bed_days_intervention_list = []

    # Iterate for the next five years
    for year in range(2022, 2028):
        # Generate baseline vector for the current year
        baseline_vector_year = generate_baseline_vector(year)
        vector_size=HF_thirty_day_readmissions(year)
        indices_to_remove = np.random.choice(vector_size, size=round(vector_size/4), replace=False)
        intervention_vector_year = np.delete(baseline_vector_year, indices_to_remove)

        # Calculate total bed days for the current year
        total_bed_days_year = np.sum(baseline_vector_year)

        # Append the total bed days to the list
        total_bed_days_list.append(total_bed_days_year)
        
        # intervention bed days total 
        total_bed_days_year_intervention = np.sum(intervention_vector_year)
        
        # append total to a list 
        
        total_bed_days_intervention_list.append(total_bed_days_year_intervention)

    # Add the total bed days list to the DataFrame
    baseline_df['Year'] = range(2022, 2028)
    baseline_df['Total_Bed_Days_Baseline'] = total_bed_days_list

    bed_day_cost=395

    baseline_df['associated cost baseline'] = baseline_df['Total_Bed_Days_Baseline']*bed_day_cost

    baseline_df['Total_Bed_Days_Intervention'] = total_bed_days_intervention_list
    baseline_df['associated cost intervention'] = baseline_df['Total_Bed_Days_Intervention']*bed_day_cost


    # Calculate cost savings per year
    baseline_df['Cost_Savings'] = baseline_df['associated cost baseline'] - baseline_df['associated cost intervention']


    
    st.markdown("### **Global and national heart failure statistics**")
    st.markdown("**Heart failure** affects at least 64 million people worldwide. It is estimated that over 900,000 people in the UK [1]. This number is likely to rise with an ageing population, more effective treatments, and improved survival rates after a heart attack. There are around 200,000 new diagnoses of heart failure in the UK every year.")
    st.markdown("Heart failure is a large burden on the NHS, accounting for **1 million bed days** per year. This makes up 2% of the NHS total, and 5% of all emergency admissions to hospital [2].")
    st.markdown("In the UK, **__18%__** of patients with HF are readmitted to hospital within 30 days [5], and the **unit cost of a heart failure hospital readmission is estimated to be £2,274** [3].")
    st.markdown("The average number of bed days for a heart failure admission is 11 days, depending on the requirement for additional specialist cardiology management[4]. According to the King's fund, the cost of a single bed day is £395.")
    st.markdown("It has been suggested that about one quarter of heart failure readmissions may be preventable. [6]")


    st.markdown("### **Heart failure in the Liverpool City region**")
    st.markdown("Around 61,000 people are living with heart and circulatory diseases.")
    st.markdown("Around 5,900 people have been diagnosed with heart failure by their GP in Liverpool.")
    st.markdown("Around 11,000 people have been diagnosed with atrial fibrillation. [2]")
   
    st.markdown("#")
    st.markdown("### The potential impact of new innovations")
    st.markdown("The figures below give a visual representation of the potential impact of a digital innovation hub for heart failure in the Liverpool city region.")
    st.markdown("Adjust the slider and hover over the plots to view the effect of an intervention (and its efficacy) on total bed days, cost to the NHS, cost savings, and return on investment.")

   
   
   
    # Create a Streamlit slider for intervention efficacy
    # Set the width of the slider
    st.markdown("""
    <style>
    .stSlider [data-baseweb=slider]{
        width: 100%;
    }
    </style>
    """,unsafe_allow_html=True)
    intervention_efficacy = st.slider("Intervention Efficacy", min_value=0, max_value=100, value=100, step = 10)

    if intervention_efficacy == 100 or intervention_efficacy== 90 or intervention_efficacy==0: 
        
        # Modify baseline_df based on the intervention efficacy
        baseline_df["Total_Bed_Days_Intervention_Adjusted"] = (
            baseline_df["Total_Bed_Days_Baseline"] + (baseline_df["Total_Bed_Days_Intervention"] - baseline_df["Total_Bed_Days_Baseline"]) * (intervention_efficacy / 100)
        )
        
    elif intervention_efficacy <=80 or intervention_efficacy>=70: 
        baseline_df["Total_Bed_Days_Intervention_Adjusted"] = baseline_df["Total_Bed_Days_Intervention_Adjusted"] = (
            baseline_df["Total_Bed_Days_Baseline"] + (baseline_df["Total_Bed_Days_Intervention"] - baseline_df["Total_Bed_Days_Baseline"]) * (intervention_efficacy / 1000)
        )
        
    else: 
        baseline_df["Total_Bed_Days_Intervention_Adjusted"] = baseline_df["Total_Bed_Days_Intervention_Adjusted"] = (
            baseline_df["Total_Bed_Days_Baseline"] + (baseline_df["Total_Bed_Days_Intervention"] - baseline_df["Total_Bed_Days_Baseline"]) * (intervention_efficacy / 100000000000000)
        )
        

    # Recalculate associated metrics based on the adjusted intervention efficacy
    baseline_df['associated cost intervention'] = baseline_df['Total_Bed_Days_Intervention_Adjusted'] * bed_day_cost
    baseline_df['Cost_Savings'] = baseline_df['associated cost baseline'] - baseline_df['associated cost intervention']

    # Line Plot
    fig = px.line(baseline_df, x="Year", y=["Total_Bed_Days_Baseline", "Total_Bed_Days_Intervention_Adjusted"],
                labels={"value": "Total Bed Days", "variable": "Line"},
                title="Total Number of Bed Days: Baseline and with intervention")

    # Create a scatter plot
    scatter_fig = px.scatter(baseline_df, x="Year", y=["Total_Bed_Days_Baseline", "Total_Bed_Days_Intervention_Adjusted"],
                        title="Total Bed Days and Intervention Bed Days Over Years")

    # Add points to the line plot
    for trace in scatter_fig.data:
        # Set showlegend=False for points
        trace.update(showlegend=False)
        fig.add_trace(trace)

    # Show the associated cost as text on hover for scatter points only
    for i, row in baseline_df.iterrows():
        for variable, cost_column in [("Total_Bed_Days_Baseline", "associated cost baseline"),
                                    ("Total_Bed_Days_Intervention_Adjusted", "associated cost intervention")]:
            cost_in_millions = row[cost_column] / 1e6  # Convert cost to millions
            cost_formatted = f"£{cost_in_millions:.1f}M"  # Format cost to one decimal place
            fig.add_trace(go.Scatter(
                x=[row["Year"]],
                y=[row[variable]],
                mode='markers',
                hoverinfo='text',
                hovertext=f"Year: {row['Year']}<br>{variable}: {row[variable]}<br>Cost:{cost_formatted}",
                marker=dict(color='black', size=0),
                showlegend=False  # Set marker size to 0 to hide the points
            ))

    # Update the legend for the main lines
    fig.update_traces(name='Total Bed Days (Baseline)', selector=dict(name='Total_Bed_Days_Baseline'))
    fig.update_traces(name='Total Bed Days (Intervention)', selector=dict(name='Total_Bed_Days_Intervention_Adjusted'))
    # Update the layout to move the legend to the bottom right
    fig.update_layout(
        legend=dict(
            orientation="h",  # Set the orientation to horizontal
            x=0.4,  # Set the x position to 1 (right)
            xanchor="left",  # Anchor the x position to the left
            y=-0.35,  # Set the y position to 0 (bottom)
            yanchor="bottom"  # Anchor the y position to the bottom
        )
    )

    # Streamlit Plot
    st.plotly_chart(fig)

    # Cost Savings Plot
    fig_cost_savings = px.bar(baseline_df, x="Year", y="Cost_Savings",
                            labels={"Cost_Savings": "Cost Savings (£M)"},
                            title="Cost Savings per Year")#,
                            #   width=400)  # Set the desired width
    st.plotly_chart(fig_cost_savings)



    from matplotlib.ticker import FuncFormatter

    # Function to calculate potential cost savings
    def calculate_savings(hospitalizations, reduction_percentage, cost_per_hospitalization):
        absolute_reduction = hospitalizations * (reduction_percentage / 100)
        cost_savings = absolute_reduction * cost_per_hospitalization
        return cost_savings


    # Function to format y-axis labels in millions
    def millions_formatter(x, pos):
        return f'£{x / 1e6:.1f}M'

    # Function to plot the visualization
    def plot_savings(hospitalizations, reduction_percentage, cost_per_hospitalization):
        savings = calculate_savings(hospitalizations, reduction_percentage, cost_per_hospitalization)

        # Plotting
        fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the figure size as needed
        ax.bar(['Current Cost', 'Potential Savings'], [hospitalizations * cost_per_hospitalization, savings], color=['#161B33', '#9E4770'])
        ax.set_ylabel('Cost (£)')

        # Adjust title size and position
        ax.set_title('Potential Cost Savings with Early Detection in Heart Failure Patients', fontsize=8, y=1.2)

        # Format y-axis labels in millions
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

        # Add labels to the bars
        for i, v in enumerate([hospitalizations * cost_per_hospitalization, savings]):
            ax.text(i, v + 0.05 * max([hospitalizations * cost_per_hospitalization, savings]), f'£{v / 1e6:.1f}M', ha='center', va='bottom')

        # Enhance aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=8)  # Adjust label size
        ax.tick_params(axis='y', labelsize=8)  # Adjust label size
        ax.grid(axis='y', linestyle='', alpha=0.7)

        fig.tight_layout()

        st.pyplot(fig)


    
    st.title('Heart Failure Cost Savings Visualisation')

    st.markdown("Evidence suggests early detection of deteriorating health in heart failure patients could reduce absolute hospitalisations by 45%. To the NHS, if 80,000 patients are hospitalised with HF, this could save approximately £50m per year.[4]")
    st.markdown("Change the parameters in the sidebar to see the impact on potential cost savings for the NHS.")
    st.markdown("#")
    # Input parameters
    st.sidebar.header('Input Parameters')
    hospitalizations = st.sidebar.number_input('Number of Hospitalizations with HF', min_value=1, step=1, value=80000)
    reduction_percentage = st.sidebar.number_input('Reduction Percentage (%)', min_value=0, max_value=100, step=1, value=45)
    cost_per_hospitalization = st.sidebar.number_input('Cost per Hospitalization (£)', min_value=1, step=1, value=1500)

    # Display input values
    st.sidebar.subheader('Input Values:')
    st.sidebar.write(f'- Number of Hospitalizations with HF: {hospitalizations}')
    st.sidebar.write(f'- Reduction Percentage: {reduction_percentage}%')
    st.sidebar.write(f'- Cost per Hospitalization: £{cost_per_hospitalization}')

    # Plotting the visualization
    plot_savings(hospitalizations, reduction_percentage, cost_per_hospitalization)


    st.markdown("### References:")

    st.markdown("[1] https://www.bhf.org.uk/-/media/files/for-professionals/research/heart-statistics/bhf-cvd-statistics-uk-factsheet.pdf")
    st.markdown("[2] https://www.bhf.org.uk/-/media/files/health-intelligence/9/liverpool-bhf-statistics.pdf")
    st.markdown("[3]https://www.nice.org.uk/guidance/ng106/resources/resource-impact-report-pdf-6537494413#")
    st.markdown("[4]https://www.england.nhs.uk/ourwork/prevention/secondary-prevention/cardiovascular-disease-high-impact-interventions/#:~:text=It%20is%20estimated%20that%20heart,into%20Living%20with%20Heart%20Failure")
    st.markdown("[5] Lawson, C., et al. (2021). Trends in 30-day readmissions following hospitalisation for heart failure by sex, socioeconomic status and ethnicity. EClinicalMedicine, 38.")
    st.markdown("[6] Khan, M. S., et al. (2021). Trends in 30-and 90-day readmission rates for heart failure. Circulation: Heart Failure, 14(4)")
   

