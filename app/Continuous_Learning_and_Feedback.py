import streamlit as st
import pandas as pd
import os
import sys
import datetime
import plotly.express as px

# Root path config
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from Continuous_learning_and_feedback.feedback import send_feedback_session_invitation
from Continuous_learning_and_feedback.alert import send_alert

# 🌙 ---- Custom GuardianAI UI Theme ----
st.markdown("""
    <style>
    body { background: linear-gradient(180deg, #0f2027, #203a43, #2c5364); color: white; }
    [data-testid="stSidebar"] { background-color: #1b2838; color: white; }
    h1, h2, h3, h4, h5 { color: #00e5ff; }
    .stButton>button { background-color: #00e5ff; color: black; border-radius: 10px; }
    .stProgress > div > div > div > div { background-color: #00e5ff; }
    </style>
""", unsafe_allow_html=True)


# ⚙️ ---- Update Police Resource Allocation ----
def update_police_allocation():
    with st.expander("🚔 **Update Police Resources**", expanded=False):
        st.subheader("Police Resource Allocation Updation")

        update_needed = st.checkbox("Do you want to update the police resource allocation?")
        if update_needed:
            data_file_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Allocation_Cleaned.csv')
            df = pd.read_csv(data_file_path)

            units = df["District Name"].unique()
            selected_unit = st.selectbox("Select the District:", units)

            current_allocation = df[df["District Name"] == selected_unit]
            current_asi = current_allocation["Sanctioned Strength of Assistant Sub-Inspectors per District"].iloc[0]
            current_chc = current_allocation["Sanctioned Strength of Head Constables per District"].iloc[0]
            current_cpc = current_allocation["Sanctioned Strength of Police Constables per District"].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Current Allocation")
                st.table(pd.DataFrame({
                    "District": [selected_unit],
                    "ASI": [current_asi],
                    "CHC": [current_chc],
                    "CPC": [current_cpc]
                }))
            with col2:
                st.subheader("✏️ Update Allocation")
                new_asi = st.number_input(f"New ASI count", min_value=int(0.7*current_asi),
                                          max_value=int(1.5*current_asi), value=int(current_asi))
                new_chc = st.number_input(f"New CHC count", min_value=int(0.7*current_chc),
                                          max_value=int(1.5*current_chc), value=int(current_chc))
                new_cpc = st.number_input(f"New CPC count", min_value=int(0.7*current_cpc),
                                          max_value=int(1.5*current_cpc), value=int(current_cpc))

            if st.button(f"✅ Confirm Update for {selected_unit}"):
                df.loc[df["District Name"] == selected_unit, [
                    "Sanctioned Strength of Assistant Sub-Inspectors per District"]] = new_asi
                df.loc[df["District Name"] == selected_unit, [
                    "Sanctioned Strength of Head Constables per District"]] = new_chc
                df.loc[df["District Name"] == selected_unit, [
                    "Sanctioned Strength of Police Constables per District"]] = new_cpc
                df.to_csv(data_file_path, index=False)
                st.success(f"✅ Police resource allocation for {selected_unit} updated successfully!")


# 🚨 ---- Alert System ----
def display_alert_meter(avg_rating, negative_feedback_count):
    with st.expander("⚠️ **Live Feedback Monitoring and Alert Meter**", expanded=True):
        rating_threshold = 3.5
        negative_feedback_threshold = 20

        rating_percentage = min(avg_rating / rating_threshold, 1.0)
        negative_feedback_percentage = min(negative_feedback_count / negative_feedback_threshold, 1.0)

        st.subheader("📈 Alert Status")
        col1, col2 = st.columns(2)
        with col1:
            st.progress(rating_percentage, text=f"Avg Rating: {avg_rating:.2f}/5")
        with col2:
            st.progress(negative_feedback_percentage, text=f"Neg. Feedback: {negative_feedback_count}/{negative_feedback_threshold}")

        if avg_rating < 3.0 or negative_feedback_count >= negative_feedback_threshold:
            st.warning("🚨 System Alert: Performance dropping. Please review the feedback and retrain models.")
            send_alert(avg_rating, rating_threshold, negative_feedback_count, negative_feedback_threshold)
        else:
            st.success("✅ System performing well based on current feedback.")

        st.markdown("*When bars reach maximum, automatic alert emails are sent to the technical lead.*")


# 🧠 ---- Continuous Learning & Feedback Main ----
def continuous_learning_and_feedback():
    st.title("🧩 GuardianAI – Continuous Learning and Feedback")
    st.caption("Enhancing predictive policing through real-time learning and user feedback.")

    update_police_allocation()

    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Feedback.csv')
    feedback_data = pd.read_csv(data_file_path)

    avg_rating = feedback_data["Feedback Rating"].mean()
    negative_feedback_count = len(feedback_data[feedback_data["Feedback Rating"] < 3])

    # 🎯 Dashboard Overview
    st.markdown("### 🔍 Overview Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rating", f"{avg_rating:.2f}/5")
    col2.metric("Negative Feedback", f"{negative_feedback_count}")
    col3.metric("Total Feedbacks", len(feedback_data))

    # 🧾 Chart Visualization
    st.markdown("### 📊 Feedback Analysis")
    fig = px.histogram(feedback_data, x="Feedback Rating", nbins=5, title="Feedback Rating Distribution",
                       color_discrete_sequence=["#00e5ff"])
    st.plotly_chart(fig, use_container_width=True)

    # 🔔 Alert Meter
    display_alert_meter(avg_rating, negative_feedback_count)

    # 🗣️ Feedback Form
    with st.expander("📝 **Provide Feedback**", expanded=False):
        feedback_form = st.form(key="feedback_form")
        feedback_type = feedback_form.selectbox("Select Feedback Type", ["Crime Pattern Analysis", "Criminal Profiling", "Predictive Modeling", "Resource Allocation"])
        feedback_rating = feedback_form.slider("Rate the system's performance", 1, 5, 3)
        feedback_comments = feedback_form.text_area("Additional Comments")
        submit_feedback = feedback_form.form_submit_button("Submit Feedback")

        if submit_feedback:
            feedback_entry = {
                "Feedback Type": feedback_type,
                "Feedback Rating": feedback_rating,
                "Feedback Comments": feedback_comments
            }
            store_feedback_data(feedback_entry)
            st.success("✅ Thank you for your feedback!")

    # 🧾 Knowledge Base
    with st.expander("📚 **Knowledge Base**", expanded=False):
        st.write("Insights and lessons learned from the continuous feedback process are documented here.")
        st.table(feedback_data.tail(10))

    # 🤝 Feedback Sessions
    with st.expander("📅 **Organize Feedback Sessions**", expanded=False):
        organize_feedback_sessions()

    # 💬 Smart Insights
    with st.expander("🤖 **AI Insights Summary**", expanded=False):
        insights = generate_insight(avg_rating, negative_feedback_count, len(feedback_data))
        st.info(insights)


# 💾 Store Feedback Data
def store_feedback_data(feedback_entry):
    file_path = os.path.join(root_dir, 'Component_datasets', 'Feedback.csv')
    try:
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([feedback_entry])
    df.to_csv(file_path, index=False)


# 🤖 Generate AI-Like Insight Summary
def generate_insight(avg_rating, neg_feedback, total_feedback):
    if avg_rating >= 4.5:
        return "🌟 Excellent performance! Users are highly satisfied with the system's accuracy."
    elif avg_rating >= 3.5:
        return f"👍 System is performing well. {total_feedback - neg_feedback} users gave positive feedback."
    elif avg_rating >= 2.5:
        return f"⚠️ System performance moderate. Retraining model might improve results."
    else:
        return "🚨 System performance critical. Review models and gather team feedback immediately."


# 👥 Manage Stakeholders
def organize_feedback_sessions():
    st.markdown("Plan and schedule collaborative feedback sessions with stakeholders.")

    session_date = st.date_input("Select Session Date")
    session_time = st.time_input("Select Session Time")

    stakeholders = get_stakeholder_contact_info()
    st.dataframe(stakeholders)

    selected_stakeholders = st.multiselect("Select Stakeholders to Invite", [s["name"] for s in stakeholders])
    if st.button("📨 Send Invitation"):
        email_list = [s["email"] for s in stakeholders if s["name"] in selected_stakeholders]
        send_feedback_session_invitation(session_date, session_time, email_list)
        st.success("✅ Invitation email sent successfully!")


def get_stakeholder_contact_info():
    return [
        {"name": "Vishal", "Position": "Technical Lead", "email": "vishalkumars.work@gmail.com"},
        {"name": "Neha", "Position": "AI Analyst", "email": "neha.ai@guardianai.in"},
        {"name": "Lucky", "Position": "System Developer", "email": "lucky.dev@guardianai.in"}
    ]
