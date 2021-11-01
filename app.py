import streamlit as st
from multiapp import MultiApp
from apps import home, basemaps, source_code, datasets


apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)
# apps.add_app("Our Work", main.app)
apps.add_app("Change basemaps", basemaps.app)
apps.add_app("Search datasets", datasets.app)
apps.add_app("Source Code", source_code.app)

# The main app
apps.run()
