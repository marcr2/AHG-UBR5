import streamlit as st

st.set_page_config(page_title="Discussion Chat", layout="centered")
st.title("💬 Hypothesis Generator vs. Critic: Discussion Chat")

# Example avatars and roles
def get_avatar_and_role(role):
    if role.lower() == "generator" or role.lower() == "hypothesis generator":
        return "🤖", "Hypothesis Generator"
    elif role.lower() == "critic":
        return "🧐", "Critic"
    else:
        return "💬", role.capitalize()

# Retrieve the conversation log from session state
conversation_log = st.session_state.get("conversation_log", [])

if not conversation_log:
    st.info("No discussion yet. Generate hypotheses to see the dialogue between the generator and critic.")
else:
    st.write("---")
    for i, entry in enumerate(conversation_log):
        # Each entry should have at least 'role' and 'content' keys
        role = entry.get("role", "generator" if i % 2 == 0 else "critic")
        content = entry.get("content") or entry.get("message") or str(entry)
        avatar, role_name = get_avatar_and_role(role)
        with st.chat_message(role_name, avatar=avatar):
            st.markdown(f"**{role_name}:** {content}") 