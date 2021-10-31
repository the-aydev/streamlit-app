import streamlit as st


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app_state = st.experimental_get_query_params()
        app_state = {
            k: v[0] if isinstance(v, list) else v for k, v in app_state.items()
        }

        titles = [a["title"] for a in self.apps]
        functions = [a["function"] for a in self.apps]
        default_radio = titles.index(app_state["page"]) if "page" in app_state else 0

        st.sidebar.title("Navigation")

        title = st.sidebar.radio("Go To", titles, index=default_radio, key="radio")

        app_state["page"] = st.session_state.radio

        st.experimental_set_query_params(**app_state)
        functions[titles.index(title)]()
