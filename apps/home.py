import streamlit as st
import leafmap.foliumap as leafmap


def app():
    st.title("OceanAI")

    st.markdown(
        """
        Make sure you set `EARTHENGINE_TOKEN='your-token'` as an environment variable (secret) on the cloud platform.
        - **Web App:** <https://share.streamlit.io/the-aydev/streamlit-app/app.py>
        - **Github:** <https://github.com/bumie-e/Saving-Our-Oceans-with-AI>
        - **Github:** <https://github.com/the-aydev/streamlit-app>
        """
    )

    with st.expander("Where to find your Earth Engine token?"):
        st.markdown(
            """
            - **Windows:** `C:/Users/USERNAME/.config/earthengine/credentials`
            """
        )

    st.subheader("Drone Imagery")
    st.markdown(
        """
        The following images were taken using drones.
    """
    )

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.image("https://camo.githubusercontent.com/b2c566839273ae50f5233625e2ba723104f6f496c90b6c748865c98b4bc93008/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d312d4b5f48313748624355386e32687476616c6a7652584d327a45576269453975")

    with row1_col2:
        st.image(
            "https://www.thebalance.com/thmb/VhBrZyc5kg8-Z0ispj-57TK5Kvw=/2121x1193/smart/filters:no_upscale()/waterpollution-ae2932fd024e4e768be14c95c5caa57c.jpg")

    st.subheader("AI Model")
    st.markdown(
        """
        The three processes are...
    """
    )

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown(
            """
        1. Data Annotation
    """
        )
        st.image("https://camo.githubusercontent.com/d2024e345fec4ca9efc1bf6ebac160fe4e57f80058f318a716c64e768b222826/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d31693850436739536f5037616c69506a677a7348397441494b6467306d58676759")
        st.markdown(
            """
        2. UNet Model Architecture.
    """
        )
        st.image("https://camo.githubusercontent.com/955335361bbcafa4dbb4b7d181eb8ae9284d365ce5db2b2042facde0f345d36f/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d31746b5f765a696253304538776b4d736838386e2d42516162395275376b504f6d")
        st.markdown(
            """
        3. Results
    """
        )
        st.image("https://camo.githubusercontent.com/2e43a3d33e39cf0e1aef060944c2a8bce2b7d2d4a1e79ffd5791de51bec1964d/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d3159664e654c6c784845516457625330563679537839316a30474c73554a39624c")

    with row1_col2:
        st.image("https://camo.githubusercontent.com/c7cbe7beead5ff55325d4f9e384c6a122b30d65fc9856d6ce2748fcad70a7e3d/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315275376b585a5059705762767036636138736b4b4e44614b644c44656f696942")
        
        st.image("https://camo.githubusercontent.com/d21856de56e75ef0f3b663d3eb0d78d808cee11fcbbda0e462e7dd4fbc0e1843/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d314e48714866776269486845644a5f7273647166615275625357522d334a6e5f4c")

    st.subheader("Impact of Our solution")
    st.markdown(
        """
        1. To Environmental Protection Agencies: Monitor the rate of garbage dumps in the oceans

        2. To Scientists: Identify regions and areas with plastic dumps

        3. To Sea Creatures: Save sea creatures life as they'll be able to breath and move freely.

    """
    )

    # Import libraries
    import ee
    # import geemap.foliumap as geemap
    import geemap.eefolium as geemap

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
