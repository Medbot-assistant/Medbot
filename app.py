import streamlit as st
from question_processing import process_question

st.title("Question Answering System")
st.write("Enter your question and get an answer from the pre-trained model.")

# Input field for the user's question
question = st.text_input("Please enter your question:")

# Process the question and display the answer(s) when the user clicks the "Submit" button
if st.button("Submit"):
    if question:
        answers = process_question(question)
        for answer in answers:
            st.write("Answer:", answer)
            st.write("---")
    else:
        st.write("Please enter a question.")
