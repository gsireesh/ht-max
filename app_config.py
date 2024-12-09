import os


BASE_CONFIG = {
    "uploaded_pdf_path": "data/uploaded_papers",
    "processed_paper_path": "data/processed_papers",
    "llm_api_keys": {},
    "mathpix_credentials": {
        "app_id": os.environ.get("MATHPIX_APP_ID", ""),
        "app_key": os.environ.get("MATHPIX_APP_KEY", ""),
    },
    "grobid_url": os.environ.get("GROBID_URL", "http://localhost:8070"),
    "chemdataextractor_service_url": os.environ.get(
        "CHEMDATAEXTRACTOR_SERVICE_URL", "http://localhost:8000"
    ),
    "matie_service_url": os.environ.get("MATIE_SERVICE_URL", "http://localhost:8003")
}

if predefined_config := os.environ.get("CONFIG_NAME"):
    if predefined_config == "sireesh_dev":
        app_config["grobid_url"] = "http://windhoek.sp.cs.cmu.edu:8070"
        app_config[
            "chemdataextractor_service_url"
        ] = "http://windhoek.sp.cs.cmu.edu:8001"
    else:  # 'docker' and everything else
        app_config["grobid_url"] = "http://collage-grobid-1:8070"
        app_config[
            "chemdataextractor_service_url"
        ] = "http://collage-chemdataextractor-1:8000"
        app_config["matie_service_url"] = "http://collage-matie-1:8000"
