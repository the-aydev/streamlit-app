import streamlit as st
from multiapp import MultiApp
from apps import home, source_code


apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)
# apps.add_app("Our Work", main.app)
apps.add_app("Source Code", source_code.app)

# The main app
apps.run()
