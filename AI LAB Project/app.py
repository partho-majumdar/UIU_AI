import streamlit as st
import numpy as np
import pickle
import time

# Load the model
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

weights = model["weights"]
mean = model["mean"]
std = model["std"]
columns = model["columns"]


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Prediction function
def predict(X, weights, threshold=0.5):
    probs = sigmoid(X.dot(weights))
    return np.array([1 if p >= threshold else 0 for p in probs])


# Streamlit page configuration
st.set_page_config(
    page_title="Term Deposit Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #818cf8;
        transform: scale(1.05);
    }
    .stNumberInput, .stSelectbox {
        background-color: #2a2a3b;
        border-radius: 8px;
        padding: 10px;
    }
    .stNumberInput input, .stSelectbox select {
        color: #ffffff;
        background-color: transparent;
        border: 1px solid #4b5563;
        border-radius: 6px;
    }
    .input-label {
        color: #d1d5db;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .result-box {
        background-color: #2a2a3b;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #4b5563;
        animation: fadeIn 0.5s ease-in;
    }
    .success-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        animation: pulseSuccess 2s infinite;
        border: none;
    }
    .failure-box {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        animation: pulseFailure 2s infinite;
        border: none;
    }
    .stat-card {
        background-color: #3b3b4b;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .stat-label {
        color: #d1d5db;
        font-size: 14px;
    }
    .stat-value {
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
    }
    .confidence-bar {
        background: #333344;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    .success-fill {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    .failure-fill {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h3 {
        color: #d1d5db;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseSuccess {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    @keyframes pulseFailure {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 15px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1>💰 Deposit Subscription Predictor</h1>
    <h3>Predict if a customer will subscribe to a term deposit with confidence</h3>
""",
    unsafe_allow_html=True,
)


with st.container():
    st.markdown("### Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="input-label">Customer Age</div>', unsafe_allow_html=True
        )
        customer_age = st.number_input(
            "",
            min_value=18,
            max_value=100,
            value=35,
            key="age",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-label">Account Balance ($)</div>', unsafe_allow_html=True
        )
        balance = st.number_input(
            "", value=1000, key="balance", label_visibility="collapsed"
        )

        st.markdown(
            '<div class="input-label">Last Contact Duration (seconds)</div>',
            unsafe_allow_html=True,
        )
        last_contact_duration = st.number_input(
            "", value=120, key="duration", label_visibility="collapsed"
        )

        st.markdown(
            '<div class="input-label">Number of Contacts in Current Campaign</div>',
            unsafe_allow_html=True,
        )
        num_contacts_in_campaign = st.number_input(
            "", min_value=1, value=2, key="contacts", label_visibility="collapsed"
        )

    with col2:
        st.markdown(
            '<div class="input-label">Previous Campaign Success?</div>',
            unsafe_allow_html=True,
        )
        prev_campaign_success = st.selectbox(
            "", ["No", "Yes"], key="prev_success", label_visibility="collapsed"
        )

        st.markdown(
            '<div class="input-label">Has Housing Loan?</div>', unsafe_allow_html=True
        )
        housing_loan_yes = st.selectbox(
            "", ["No", "Yes"], key="housing_loan", label_visibility="collapsed"
        )

        st.markdown(
            '<div class="input-label">Has Personal Loan?</div>', unsafe_allow_html=True
        )
        personal_loan_yes = st.selectbox(
            "", ["No", "Yes"], key="personal_loan", label_visibility="collapsed"
        )

        st.markdown(
            '<div class="input-label">Has Tertiary Education?</div>',
            unsafe_allow_html=True,
        )
        education_tertiary = st.selectbox(
            "", ["No", "Yes"], key="education", label_visibility="collapsed"
        )


user_input = np.array(
    [
        customer_age,
        balance,
        last_contact_duration,
        num_contacts_in_campaign,
        1 if prev_campaign_success == "Yes" else 0,
        1 if housing_loan_yes == "Yes" else 0,
        1 if personal_loan_yes == "Yes" else 0,
        1 if education_tertiary == "Yes" else 0,
    ],
    dtype=float,
).reshape(1, -1)

X_new = (user_input - mean) / std
X_new = np.c_[np.ones(X_new.shape[0]), X_new]


if st.button("🔮 Predict Subscription"):
    with st.spinner("🤖 Analyzing customer data..."):
        time.sleep(1.5)

        prediction = predict(X_new, weights)[0]
        probability = sigmoid(X_new.dot(weights))[0]

    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    progress_bar.empty()

    if prediction == 1:
        st.markdown(
            f"""
            <div class="success-box">
                <div style="text-align: center;">
                    <h2 style="margin: 0; font-size: 2em;">🎉 HIGH POTENTIAL CUSTOMER!</h2>
                    <p style="font-size: 1.2em; margin: 10px 0;">This customer <strong>WILL SUBSCRIBE</strong> to a term deposit</p>
                    <div style="font-size: 1.5em; font-weight: bold; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; display: inline-block;">
                        {probability:.1%} Confidence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="color: #d1d5db; margin-bottom: 5px;">Confidence Level:</div>
            <div class="confidence-bar">
                <div class="confidence-fill success-fill" style="width: {probability*100}%;"></div>
            </div>
            <div style="text-align: center; color: #10b981; font-weight: bold; margin-top: 5px;">
                {probability:.1%} - High Conversion Probability
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("💡 Recommended Actions", expanded=True):
            st.markdown(
                """
            - **Immediate Follow-up**: Contact within 24 hours
            - **Personalized Offer**: Tailor terms to customer profile  
            - **Priority Handling**: Assign to experienced agent
            - **Cross-sell**: Explore additional banking products
            """
            )

    else:
        st.markdown(
            f"""
            <div class="failure-box">
                <div style="text-align: center;">
                    <h2 style="margin: 0; font-size: 2em;">⚠️ LOW CONVERSION PROBABILITY</h2>
                    <p style="font-size: 1.2em; margin: 10px 0;">This customer <strong>WILL NOT SUBSCRIBE</strong> to a term deposit</p>
                    <div style="font-size: 1.5em; font-weight: bold; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; display: inline-block;">
                        {(1-probability):.1%} Confidence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="color: #d1d5db; margin-bottom: 5px;">Confidence Level:</div>
            <div class="confidence-bar">
                <div class="confidence-fill failure-fill" style="width: {(1-probability)*100}%;"></div>
            </div>
            <div style="text-align: center; color: #ef4444; font-weight: bold; margin-top: 5px;">
                {(1-probability):.1%} - Low Conversion Probability
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Alternative strategies
        # with st.expander("🔄 Alternative Strategies", expanded=True):
        #     st.markdown(
        #         """
        #     - **Re-engagement Campaign**: Follow up in 30-60 days
        #     - **Different Products**: Suggest savings or investment accounts
        #     - **Educational Content**: Provide financial planning resources
        #     - **Segment Analysis**: May respond to different marketing approach
        #     """
        #     )

    # Display key statistics
    st.markdown("### 📊 Prediction Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Prediction Confidence</div>
                <div class="stat-value">{max(probability, 1-probability):.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Input Features</div>
                <div class="stat-value">{len(user_input[0])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        status = "High Potential" if prediction == 1 else "Needs Strategy"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Customer Status</div>
                <div class="stat-value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """

    """,
    unsafe_allow_html=True,
)
