import streamlit as st
import leafmap.foliumap as leafmap


def app():
    st.title("OceanAI")

    st.markdown(
        """
        This multi-page web app show the use of OceanAI.
        This site demostrates how to build a multi-page [Earth Engine](https://earthengine.google.com) App using [streamlit](https://streamlit.io) and [geemap](https://geemap.org).
        You can deploy the app on various cloud platforms, such as [share.streamlit.io](https://share.streamlit.io) or [Heroku](https://heroku.com).
        Make sure you set `EARTHENGINE_TOKEN='your-token'` as an environment variable (secret) on the cloud platform.
        - **Web App:** <https://share.streamlit.io/the-aydev/streamlit-app/app.py>
        - **Github:** <https://github.com/the-aydev/streamlit-app>
        """
    )

    with st.expander("Where to find your Earth Engine token?"):
        st.markdown(
            """
            - **Windows:** `C:/Users/USERNAME/.config/earthengine/credentials`
            """
        )

    st.subheader("Satellite Imagery")
    st.markdown(
        """
        The following images were taken using drones.
    """
    )

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.image("https://i.imgur.com/NRcS93r.gif")
        st.image("https://i.imgur.com/TwO7AGE.gif")

    with row1_col2:
        st.image("https://i.imgur.com/e0iEq3N.gif")
        st.image("https://i.imgur.com/WlSNoGt.gif")

    # Import libraries
    import ee
    import geemap.foliumap as geemap

    # Create an interactive map
    Map = geemap.Map(plugin_Draw=True, Draw_export=False)
    # Add a basemap
    Map.add_basemap("TERRAIN")
    # Retrieve Earth Engine dataset
    dem = ee.Image("USGS/SRTMGL1_003")
    # Set visualization parameters
    vis_params = {
        "min": 0,
        "max": 4000,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
    }
    # Add the Earth Engine image to the map
    Map.addLayer(dem, vis_params, "SRTM DEM", True, 0.5)
    # Add a colorbar to the map
    Map.add_colorbar(vis_params["palette"], 0, 4000, caption="Elevation (m)")
    # Render the map using streamlit
    Map.to_streamlit()