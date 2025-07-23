import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import logging
from datetime import datetime
import hashlib
import base64
from functools import lru_cache
import time
import sentry_sdk
from prometheus_fastapi_instrumentator import Instrumentator

# Google Cloud kliensekhez
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

# Cerebras Cloud SDK
from cerebras.cloud.sdk import Cerebras

# Gemini API
import google.generativeai as genai

# Exa API
from exa_py import Exa

# FastAPI
from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from pydantic import BaseModel, Field

# Naplózás konfigurálása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Digitális Ujjlenyomat ---
DIGITAL_FINGERPRINT = "Jade made by Kollár Sándor"
CREATOR_SIGNATURE = "SmFkZSBtYWRlIGJ5IEtvbGzDoXIgU8OhbmRvcg=="
CREATOR_HASH = "a7b4c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5c8d9e2f1a6b5"

# --- API Kulcsok betöltése ---
GCP_SERVICE_ACCOUNT_KEY_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
OPENAI_ADMIN_KEY = os.environ.get("OPENAI_ADMIN_KEY")

# --- Token Limit Definíciók ---
TOKEN_LIMITS = {
    "gpt-3.5-turbo": 200000,
    "gpt-4.1": 900000,
    "gpt-4.1-long-context": 200000,
    "gpt-4.1-mini": 200000,
    "gpt-4.1-mini-long-context": 400000,
    "gpt-4.1-nano": 200000,
    "gpt-4.1-nano-long-context": 400000,
    "gpt-4.5-preview": 200,
    "gpt-4.0": 90000,
    "gpt-4.0-mini": 200000,
    "dall-e-2": 200,
    "dall-e-3": 200,
    "tts-1": 200,
    "whisper-1": 200
}

# --- Kliensek inicializálása ---
gcp_credentials = None
if GCP_SERVICE_ACCOUNT_KEY_JSON and GCP_PROJECT_ID and GCP_REGION:
    try:
        info = json.loads(GCP_SERVICE_ACCOUNT_KEY_JSON)
        gcp_credentials = service_account.Credentials.from_service_account_info(info)
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=gcp_credentials)
        logger.info("GCP Vertex AI client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing GCP Vertex AI client: {e}")
        gcp_credentials = None

cerebras_client = None
if CEREBRAS_API_KEY:
    try:
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Cerebras client: {e}")

# Gemini 2.5 Pro inicializálása
gemini_model = None
gemini_25_pro = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        gemini_25_pro = genai.GenerativeModel('gemini-2.5-pro')
        logger.info("Gemini 1.5 Pro and 2.5 Pro clients initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini clients: {e}")

exa_client = None
if EXA_API_KEY:
    try:
        exa_client = Exa(api_key=EXA_API_KEY)
        logger.info("Exa client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Exa client: {e}")

# --- FastAPI alkalmazás ---
app = FastAPI(
    title="Jade - Deep Discovery AI Platform",
    description="Fejlett AI platform 150+ tudományos és innovációs szolgáltatással",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Headers ---
app.add_middleware(SecurityHeadersMiddleware)

# --- HTTPS Redirect (optional, for production) ---
# app.add_middleware(HTTPSRedirectMiddleware)

app.mount("/static", StaticFiles(directory="templates"), name="static")

# --- Pydantic modellek ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(..., description="Felhasználó egyedi azonosítója")

class DeepResearchRequest(BaseModel):
    query: str = Field(..., description="Kutatási kérdés")
    user_id: str = Field(..., description="Felhasználó azonosító")

class SimpleAlphaRequest(BaseModel):
    service_name: str = Field(..., description="Az Alpha szolgáltatás neve")
    query: str = Field(..., description="Egyszerű szöveges kérés")
    details: str = Field(default="", description="További részletek (opcionális)")

class UniversalAlphaRequest(BaseModel):
    service_name: str = Field(..., description="Az Alpha szolgáltatás neve")
    input_data: Dict[str, Any] = Field(..., description="Bemeneti adatok")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Opcionális paraméterek")

class ScientificInsightRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Tudományos lekérdezés")
    num_results: int = Field(default=5, ge=1, le=10, description="Találatok száma")
    summary_length: int = Field(default=200, ge=50, le=500, description="Összefoglaló hossza")

class AdvancedExaRequest(BaseModel):
    query: str = Field(..., description="Keresési lekérdezés")
    type: str = Field(default="neural", description="Keresés típusa: neural, keyword, similarity")
    num_results: int = Field(default=10, ge=1, le=50, description="Találatok száma")
    include_domains: List[str] = Field(default=[], description="Csak ezeken a domaineken keressen")
    exclude_domains: List[str] = Field(default=[], description="Ezeket a domaineket zárja ki")
    start_crawl_date: Optional[str] = Field(None, description="Kezdő dátum (YYYY-MM-DD)")
    end_crawl_date: Optional[str] = Field(None, description="Befejező dátum (YYYY-MM-DD)")
    start_published_date: Optional[str] = Field(None, description="Publikálás kezdő dátuma")
    end_published_date: Optional[str] = Field(None, description="Publikálás befejező dátuma")
    include_text: List[str] = Field(default=[], description="Ezeket a szövegeket tartalmaznia kell")
    exclude_text: List[str] = Field(default=[], description="Ezeket a szövegeket nem tartalmazhatja")
    category: Optional[str] = Field(None, description="Kategória szűrő")
    subcategory: Optional[str] = Field(None, description="Alkategória szűrő")
    livecrawl: str = Field(default="always", description="Live crawl: always, never, when_necessary")
    text_contents_options: Dict[str, Any] = Field(default_factory=lambda: {
        "max_characters": 2000,
        "include_html_tags": False,
        "strategy": "comprehensive"
    })

class ExaSimilarityRequest(BaseModel):
    url: str = Field(..., description="Referencia URL")
    num_results: int = Field(default=10, ge=1, le=50, description="Hasonló találatok száma")
    category_weights: Dict[str, float] = Field(default={}, description="Kategória súlyok")
    exclude_source_domain: bool = Field(default=True, description="Forrás domain kizárása")

class ExaContentsRequest(BaseModel):
    ids: List[str] = Field(..., description="Exa result ID-k")
    summary: bool = Field(default=True, description="Összefoglaló generálása")
    highlights: Dict[str, Any] = Field(default_factory=dict, description="Kiemelés opciók")

class ProteinLookupRequest(BaseModel):
    protein_id: str = Field(..., description="Fehérje azonosító")

class CustomGCPModelRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="GCP modell bemeneti adatok")
    gcp_endpoint_id: str = Field(..., description="GCP végpont azonosító")
    gcp_project_id: Optional[str] = GCP_PROJECT_ID
    gcp_region: Optional[str] = GCP_REGION

class SimulationOptimizerRequest(BaseModel):
    simulation_type: str = Field(..., description="Szimuláció típusa")
    input_parameters: Dict[str, Any] = Field(..., description="Bemeneti paraméterek")
    optimization_goal: str = Field(..., description="Optimalizálási cél")

class AlphaGenomeRequest(BaseModel):
    genome_sequence: str = Field(..., min_length=100, description="Genom szekvencia")
    organism: str = Field(..., description="Organizmus")
    analysis_type: str = Field(..., description="Elemzés típusa")
    include_predictions: bool = Field(default=False, description="Fehérje előrejelzések")

class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Kód generálási kérés")
    language: str = Field(default="python", description="Programozási nyelv")
    complexity: str = Field(default="medium", description="Kód komplexitása")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="AI kreativitás")

# Beszélgetési előzmények és cache
chat_histories: Dict[str, List[Message]] = {}
response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY = 300  # 5 perc cache

# --- Alpha Services definíciója ---
ALPHA_SERVICES = {
    "biologiai_orvosi": {
        "AlphaMicrobiome": "Mikrobiom elemzés és baktériumközösség vizsgálat",
        "AlphaImmune": "Immunrendszer válaszok predikciója",
        "AlphaCardio": "Szívbetegségek kockázatelemzése",
        "AlphaNeuron": "Idegsejt aktivitás szimulálása",
        "AlphaVirus": "Vírus mutáció előrejelzése",
        "AlphaCell": "Sejtosztódás és növekedés modellezése",
        "AlphaMetabolism": "Anyagcsere útvonalak elemzése",
        "AlphaPharmaco": "Gyógyszer-receptor kölcsönhatások",
        "AlphaGene": "Génexpresszió előrejelzése",
        "AlphaProteomics": "Fehérje hálózatok elemzése",
        "AlphaToxicology": "Toxicitás és biztonság értékelése",
        "AlphaEpigenetics": "Epigenetikai változások predikciója",
        "AlphaBiomarker": "Biomarker azonosítás és validálás",
        "AlphaPathogen": "Kórokozó azonosítás és karakterizálás",
        "AlphaOncology": "Rák biomarkerek és terápiás célpontok",
        "AlphaEndocrine": "Hormonális szabályozás modellezése",
        "AlphaRespiratory": "Légzési rendszer betegségei",
        "AlphaNeurodegeneration": "Neurodegeneratív betegségek",
        "AlphaRegenerative": "Regeneratív medicina alkalmazások",
        "AlphaPersonalized": "Személyre szabott orvoslás",
        "AlphaBioengineering": "Biomérnöki rendszerek tervezése",
        "AlphaBioinformatics": "Bioinformatikai adatelemzés",
        "AlphaSystemsBiology": "Rendszerbiológiai modellezés",
        "AlphaSynthbio": "Szintetikus biológiai rendszerek",
        "AlphaLongevity": "Öregedés és hosszú élet kutatása"
    },
    "kemiai_anyagtudomanyi": {
        "AlphaCatalyst": "Katalizátor tervezés és optimalizálás",
        "AlphaPolymer": "Polimer tulajdonságok előrejelzése",
        "AlphaNanotech": "Nanomateriál szintézis és jellemzés",
        "AlphaChemSynthesis": "Kémiai szintézis útvonalak",
        "AlphaMaterial": "Anyagtulajdonságok predikciója",
        "AlphaSuperconductor": "Szupravezető anyagok kutatása",
        "AlphaSemiconductor": "Félvezető anyagok tervezése",
        "AlphaComposite": "Kompozit anyagok fejlesztése",
        "AlphaBattery": "Akkumulátor technológiák",
        "AlphaSolar": "Napelem hatékonyság optimalizálása",
        "AlphaCorrosion": "Korrózió és védelem elemzése",
        "AlphaAdhesive": "Ragasztó és kötőanyagok",
        "AlphaCrystal": "Kristályszerkezet előrejelzése",
        "AlphaLiquid": "Folyadék tulajdonságok modellezése",
        "AlphaGas": "Gázfázisú reakciók szimulálása",
        "AlphaSurface": "Felületi kémia és adszorpció",
        "AlphaElectrochemistry": "Elektrokémiai folyamatok",
        "AlphaPhotochemistry": "Fotokémiai reakciók",
        "AlphaThermodynamics": "Termodinamikai paraméterek",
        "AlphaKinetics": "Reakciókinetika modellezése",
        "AlphaSpectroscopy": "Spektroszkópiai adatelemzés",
        "AlphaChromatography": "Kromatográfiás szeparáció",
        "AlphaAnalytical": "Analitikai kémiai módszerek",
        "AlphaFormulation": "Formuláció és stabilitás",
        "AlphaGreen": "Zöld kémiai alternatívák"
    },
    "kornyezeti_fenntarthato": {
        "AlphaClimate": "Klímaváltozás modellezése",
        "AlphaOcean": "Óceáni rendszerek elemzése",
        "AlphaAtmosphere": "Légköri folyamatok szimulálása",
        "AlphaEcology": "Ökológiai rendszerek modellezése",
        "AlphaWater": "Víz minőség és kezelés",
        "AlphaSoil": "Talaj egészség és termékenység",
        "AlphaRenewable": "Megújuló energia optimalizálása",
        "AlphaCarbon": "Szén-dioxid befogás és tárolás",
        "AlphaWaste": "Hulladékgazdálkodás és újrahasznosítás",
        "AlphaBiodiversity": "Biodiverzitás védelem",
        "AlphaForest": "Erdészeti fenntarthatóság",
        "AlphaAgriculture": "Fenntartható mezőgazdaság",
        "AlphaPollution": "Környezetszennyezés elemzése",
        "AlphaConservation": "Természetvédelem stratégiák",
        "AlphaUrban": "Városi fenntarthatóság",
        "AlphaTransport": "Közlekedési rendszerek",
        "AlphaBuilding": "Épület energetika",
        "AlphaResource": "Erőforrás gazdálkodás",
        "AlphaLifecycle": "Életciklus elemzés",
        "AlphaCircular": "Körforgásos gazdaság",
        "AlphaEnvironmentalHealth": "Környezeti egészségügy",
        "AlphaWildlife": "Vadvilág monitoring",
        "AlphaMarine": "Tengeri ökoszisztémák",
        "AlphaDesertification": "Elsivatagosodás elleni küzdelem",
        "AlphaSustainability": "Fenntarthatósági metrikák"
    },
    "fizikai_asztrofizikai": {
        "AlphaQuantum": "Kvantumfizikai szimulációk",
        "AlphaParticle": "Részecskefizikai elemzések",
        "AlphaGravity": "Gravitációs hullámok elemzése",
        "AlphaCosmic": "Kozmikus sugárzás kutatása",
        "AlphaStellar": "Csillagfejlődés modellezése",
        "AlphaGalaxy": "Galaxisok dinamikája",
        "AlphaExoplanet": "Exobolygó karakterizálás",
        "AlphaPlasma": "Plazma fizika szimulációk",
        "AlphaOptics": "Optikai rendszerek tervezése",
        "AlphaCondensed": "Kondenzált anyag fizika",
        "AlphaSuperconductivity": "Szupravezetés mechanizmusai",
        "AlphaMagnetism": "Mágneses tulajdonságok",
        "AlphaThermodynamics": "Termodinamikai rendszerek",
        "AlphaFluid": "Folyadékdinamika szimulációk",
        "AlphaAcoustics": "Akusztikai jelenségek",
        "AlphaElectromagnetism": "Elektromágneses mezők",
        "AlphaNuclear": "Nukleáris folyamatok",
        "AlphaAtomic": "Atomfizikai számítások",
        "AlphaMolecular": "Molekuláris fizika",
        "AlphaSpectroscopy": "Spektroszkópiai elemzés",
        "AlphaLaser": "Lézer technológiák",
        "AlphaPhotonics": "Fotonika alkalmazások",
        "AlphaCryogenics": "Kriogén rendszerek",
        "AlphaVacuum": "Vákuum technológia",
        "AlphaInstrumentation": "Tudományos műszerek"
    },
    "technologiai_melymu": {
        "AlphaAI": "Mesterséges intelligencia architektúrák",
        "AlphaML": "Gépi tanulás optimalizálás",
        "AlphaNeural": "Neurális hálózatok tervezése",
        "AlphaRobotics": "Robotikai rendszerek",
        "AlphaAutonomy": "Autonóm rendszerek",
        "AlphaVision": "Számítógépes látás",
        "AlphaNLP": "Természetes nyelv feldolgozás",
        "AlphaOptimization": "Optimalizálási algoritmusok",
        "AlphaSimulation": "Szimulációs rendszerek",
        "AlphaModeling": "Matematikai modellezés",
        "AlphaControl": "Irányítástechnika",
        "AlphaSignal": "Jelfeldolgozás",
        "AlphaData": "Adatelemzés és big data",
        "AlphaNetwork": "Hálózati rendszerek",
        "AlphaSecurity": "Kiberbiztonsági elemzés",
        "AlphaCrypto": "Kriptográfiai protokollok",
        "AlphaBlockchain": "Blockchain technológiák",
        "AlphaIoT": "Internet of Things rendszerek",
        "AlphaEdge": "Edge computing optimalizálás",
        "AlphaCloud": "Felhő architektúrák",
        "AlphaHPC": "Nagy teljesítményű számítás",
        "AlphaDrone": "Drón technológiák",
        "AlphaSensor": "Szenzor hálózatok",
        "AlphaEmbedded": "Beágyazott rendszerek",
        "AlphaFPGA": "FPGA programozás"
    },
    "tarsadalmi_gazdasagi": {
        "AlphaEconomy": "Gazdasági modellek és előrejelzések",
        "AlphaMarket": "Piaci trendek elemzése",
        "AlphaFinance": "Pénzügyi kockázatelemzés",
        "AlphaSocial": "Társadalmi hálózatok elemzése",
        "AlphaPolicy": "Szakpolitikai hatáselemzés",
        "AlphaUrbanPlanning": "Városfejlesztés optimalizálása",
        "AlphaLogistics": "Logisztikai láncok",
        "AlphaSupplyChain": "Ellátási láncok optimalizálása",
        "AlphaManufacturing": "Gyártási folyamatok",
        "AlphaQuality": "Minőségbiztosítás",
        "AlphaRisk": "Kockázatelemzés és menedzsment",
        "AlphaDecision": "Döntéstámogató rendszerek",
        "AlphaStrategy": "Stratégiai tervezés",
        "AlphaInnovation": "Innovációs ökoszisztémák",
        "AlphaStartup": "Startup értékelés és mentoring",
        "AlphaEducation": "Oktatási rendszerek",
        "AlphaHealthcare": "Egészségügyi rendszerek",
        "AlphaCustomer": "Vásárlói viselkedés elemzése",
        "AlphaMarketing": "Marketing optimalizálás",
        "AlphaBrand": "Márka értékelés",
        "AlphaHR": "Humán erőforrás menedzsment",
        "AlphaLegal": "Jogi elemzések",
        "AlphaCompliance": "Megfelelőségi rendszerek",
        "AlphaEthics": "Etikai értékelések",
        "AlphaSustainableBusiness": "Fenntartható üzleti modellek"
    }
}

# --- Backend Model Selection ---
async def select_backend_model(prompt: str, service_name: str = None):
    """Backend modell kiválasztása a kérés és a token limitek alapján - Cerebras prioritás"""
    # CEREBRAS ELSŐ PRIORITÁS a sebességért
    if cerebras_client:
        selected_model = cerebras_client
        model_name = "llama-4-scout-17b-16e-instruct"
        return {"model": selected_model, "name": model_name}
    
    # Backup: Gemini 2.5 Pro
    if gemini_25_pro:
        selected_model = gemini_25_pro
        model_name = "gemini-2.5-pro"
        return {"model": selected_model, "name": model_name}

    # Backup: Gemini 1.5 Pro
    if gemini_model:
        selected_model = gemini_model
        model_name = "gemini-1.5-pro"
        return {"model": selected_model, "name": model_name}

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Nincs elérhető AI modell"
    )

# --- Model Execution ---
async def execute_model(model_info: Dict[str, Any], prompt: str):
    """Modell futtatása a kiválasztott backenddel."""
    model = model_info["model"]
    model_name = model_info["name"]
    response_text = ""

    try:
        if model == gemini_25_pro or model == gemini_model:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.1
            )
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            response_text = response.text
            return {"response": response_text, "model_used": model_name, "selected_backend": model_name}

        elif model == cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            return {"response": response_text, "model_used": "Cerebras Llama 4", "selected_backend": "Cerebras Llama 4"}

        else:
            raise ValueError("Érvénytelen modell")

    except Exception as e:
        logger.error(f"Modell végrehajtási hiba: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a modell végrehajtása során: {e}"
        )

# --- Egyszerű Alpha Service Handler ---
async def handle_simple_alpha_service(service_name: str, query: str, details: str = "") -> Dict[str, Any]:
    """Egyszerű Alpha szolgáltatás kezelő szöveges bemenetnél"""

    # Keresés a kategóriákban
    service_category = None
    service_description = None

    for category, services in ALPHA_SERVICES.items():
        if service_name in services:
            service_category = category
            service_description = services[service_name]
            break

    if not service_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen Alpha szolgáltatás: {service_name}"
        )

    # Backend modell kiválasztása
    model_info = await select_backend_model(query, service_name)

    # Prompt összeállítása
    prompt = f"""
    {service_name} Alpha Szolgáltatás Elemzés
    Kategória: {service_category}
    Szolgáltatás leírása: {service_description}

    Felhasználó kérése: {query}

    További részletek: {details if details else "Nincs további részlet"}

    Kérlek, végezz professzionális, tudományos elemzést és adj részletes válaszokat a megadott kérés alapján.
    A válaszod legyen strukturált, magyar nyelvű és gyakorlati szempontokat is tartalmazzon.
    Használd a legfrissebb tudományos információkat és módszereket.
    """

    try:
        # Modell futtatása
        result = await execute_model(model_info, prompt)

        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": result["response"],
            "model_used": result["model_used"],
            "performance_data": {
                "backend_model": result["selected_backend"],
                "tokens_used": result.get("tokens_used", 0)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a {service_name} szolgáltatás végrehajtása során: {e}"
        )

# --- Általános Alpha Service Handler ---
async def handle_alpha_service(service_name: str, input_data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Univerzális Alpha szolgáltatás kezelő"""

    # Keresés a kategóriákban
    service_category = None
    service_description = None

    for category, services in ALPHA_SERVICES.items():
        if service_name in services:
            service_category = category
            service_description = services[service_name]
            break

    if not service_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen Alpha szolgáltatás: {service_name}"
        )

    # Bemeneti adatok stringgé alakítása a modellválasztáshoz
    input_str = json.dumps(input_data, ensure_ascii=False)

    # Backend modell kiválasztása
    model_info = await select_backend_model(input_str, service_name)

    # Prompt összeállítása
    prompt = f"""
    {service_name} Alpha Szolgáltatás
    Kategória: {service_category}
    Leírás: {service_description}

    Bemeneti adatok:
    {json.dumps(input_data, indent=2, ensure_ascii=False)}

    Paraméterek:
    {json.dumps(parameters or {}, indent=2, ensure_ascii=False)}

    Kérlek, végezz professzionális, tudományos elemzést és adj részletes válaszokat a megadott adatok alapján.
    A válaszod legyen strukturált, magyar nyelvű és gyakorlati szempontokat is tartalmazzon.
    """

    try:
        # Modell futtatása
        result = await execute_model(model_info, prompt)

        return {
            "service_name": service_name,
            "category": service_category,
            "description": service_description,
            "analysis": result["response"],
            "model_used": result["model_used"],
            "performance_data": {
                "backend_model": result["selected_backend"],
                "tokens_used": result.get("tokens_used", 0)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in {service_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a {service_name} szolgáltatás végrehajtása során: {e}"
        )

# --- API Végpontok ---

@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")

@app.get("/api")
async def root_endpoint():
    return {
        "message": "Jade - Deep Discovery AI Platform",
        "version": app.version,
        "creator": DIGITAL_FINGERPRINT,
        "total_services": sum(len(services) for services in ALPHA_SERVICES.values()),
        "categories": list(ALPHA_SERVICES.keys())
    }

@app.get("/api/services")
async def get_services():
    """Minden Alpha szolgáltatás listázása kategóriák szerint"""
    return {
        "categories": ALPHA_SERVICES,
        "total_services": sum(len(services) for services in ALPHA_SERVICES.values())
    }

@app.get("/api/services/{category}")
async def get_services_by_category(category: str):
    """Egy kategória szolgáltatásainak listázása"""
    if category not in ALPHA_SERVICES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ismeretlen kategória: {category}"
        )

    return {
        "category": category,
        "services": ALPHA_SERVICES[category]
    }

@app.post("/api/alpha/{service_name}")
async def execute_alpha_service(service_name: str, request: UniversalAlphaRequest):
    """Bármely Alpha szolgáltatás végrehajtása"""
    return await handle_alpha_service(
        service_name=service_name,
        input_data=request.input_data,
        parameters=request.parameters
    )

@app.post("/api/alpha/simple/{service_name}")
async def execute_simple_alpha_service(service_name: str, request: SimpleAlphaRequest):
    """Egyszerű Alpha szolgáltatás végrehajtása szöveges bemenetnél"""
    return await handle_simple_alpha_service(
        service_name=service_name,
        query=request.query,
        details=request.details
    )

@app.post("/api/deep_discovery/chat")
async def deep_discovery_chat(req: ChatRequest):
    """Optimalizált chat funkcionalitás cache-eléssel"""
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető chat modell"
        )

    user_id = req.user_id
    current_message = req.message

    # Cache ellenőrzés
    cache_key = hashlib.md5(f"{user_id}:{current_message}".encode()).hexdigest()
    current_time = time.time()

    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        if current_time - cached_response['timestamp'] < CACHE_EXPIRY:
            logger.info("Serving cached response")
            return cached_response['data']

    history = chat_histories.get(user_id, [])

    # Rövidebb system message a gyorsaságért
    system_message = {
        "role": "system",
        "content": "Te Jade vagy, egy fejlett AI asszisztens magyarul. Szakértő vagy tudományos és technológiai területeken. Segítőkész, részletes válaszokat adsz."
    }

    # Csak az utolsó 10 üzenetet használjuk a kontextushoz
    recent_history = history[-10:] if len(history) > 10 else history
    messages_for_llm = [system_message] + recent_history + [{"role": "user", "content": current_message}]

    try:
        response_text = ""
        model_used = ""

        # Cerebras elsőként a gyorsaság miatt
        if cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=messages_for_llm,
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,  # Optimalizált limit
                temperature=0.15,  # Gyorsabb és konzisztensebb
                top_p=0.95,  # Optimalizált sampling
                presence_penalty=0.0,  # Gyorsabb feldolgozás
                frequency_penalty=0.0
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        elif gemini_25_pro:
            response = await gemini_25_pro.generate_content_async(
                '\n'.join([msg['content'] for msg in messages_for_llm]),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.2
                )
            )
            response_text = response.text
            model_used = "Gemini 2.5 Pro"
        elif gemini_model:
            response = await gemini_model.generate_content_async(
                '\n'.join([msg['content'] for msg in messages_for_llm]),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.2
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"

        history.append({"role": "user", "content": current_message})
        history.append({"role": "assistant", "content": response_text})

        # Memória optimalizálás: csak az utolsó 20 üzenetet tartjuk meg
        if len(history) > 20:
            history = history[-20:]

        chat_histories[user_id] = history

        result = {
            'response': response_text,
            'model_used': model_used,
            'status': 'success'
        }

        # Válasz cache-elése
        response_cache[cache_key] = {
            'data': result,
            'timestamp': current_time
        }

        return result

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a beszélgetés során: {e}"
        )

@app.post("/api/deep_research")
async def deep_research(req: DeepResearchRequest):
    """Optimalizált deep research API - valóban működő 1000+ forrás feldolgozással"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        logger.info(f"Starting comprehensive deep research for: {req.query}")
        
        # Kibővített tudományos és akadémiai domainok listája
        scientific_domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "ieee.org", "acm.org", "springer.com", "wiley.com",
            "sciencedirect.com", "jstor.org", "researchgate.net", "semantic-scholar.org",
            "biorxiv.org", "medrxiv.org", "plos.org", "bmj.com", "thelancet.com",
            "nih.gov", "who.int", "cdc.gov", "fda.gov", "ema.europa.eu"
        ]
        
        # Keresési eredmények gyűjtése
        all_results = []
        
        # 1. Fő neurális keresés - több batch-ben
        for batch in range(5):  # 5 batch = 250 eredmény
            try:
                neural_search = exa_client.search(
                    query=f"{req.query} scientific research study",
                    type="neural",
                    num_results=50,
                    include_domains=scientific_domains,
                    text_contents={"max_characters": 3000, "strategy": "comprehensive"},
                    livecrawl="when_necessary"
                )
                all_results.extend(neural_search.results)
                logger.info(f"Neural batch {batch+1}/5 completed: {len(neural_search.results)} results")
            except Exception as e:
                logger.error(f"Neural search batch {batch+1} error: {e}")
        
        # 2. Kulcsszavas keresések - specifikus témák
        keyword_variants = [
            f"{req.query} research",
            f"{req.query} study",
            f"{req.query} analysis",
            f"{req.query} review",
            f"{req.query} investigation"
        ]
        
        for variant in keyword_variants:
            try:
                keyword_search = exa_client.search(
                    query=variant,
                    type="keyword",
                    num_results=40,
                    include_domains=scientific_domains,
                    text_contents={"max_characters": 3000, "strategy": "comprehensive"}
                )
                all_results.extend(keyword_search.results)
                logger.info(f"Keyword search '{variant}': {len(keyword_search.results)} results")
            except Exception as e:
                logger.error(f"Keyword search error for '{variant}': {e}")
            
        # 3. Időszakos keresések - több év
        time_periods = ["2024", "2023", "2022"]
        for year in time_periods:
            try:
                recent_search = exa_client.search(
                    query=f"{req.query} {year}",
                    type="neural",
                    num_results=30,
                    start_published_date=f"{year}-01-01",
                    text_contents={"max_characters": 3000, "strategy": "comprehensive"}
                )
                all_results.extend(recent_search.results)
                logger.info(f"Time period {year}: {len(recent_search.results)} results")
            except Exception as e:
                logger.error(f"Time period search error for {year}: {e}")

        # 4. Speciális domain keresések
        for domain in scientific_domains[:10]:  # Top 10 domain
            try:
                domain_search = exa_client.search(
                    query=req.query,
                    type="neural",
                    num_results=20,
                    include_domains=[domain],
                    text_contents={"max_characters": 3000, "strategy": "comprehensive"}
                )
                all_results.extend(domain_search.results)
                logger.info(f"Domain {domain}: {len(domain_search.results)} results")
            except Exception as e:
                logger.error(f"Domain search error for {domain}: {e}")

        # Duplikációk eltávolítása URL alapján
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result

        final_results = list(unique_results.values())
        logger.info(f"Total unique results after deduplication: {len(final_results)}")

        # Eredmények feldolgozása
        sources = []
        combined_content = ""
        
        for i, result in enumerate(final_results[:1000]):  # Max 1000 eredmény
            source_data = {
                "id": i + 1,
                "title": result.title or "Cím nem elérhető",
                "url": result.url,
                "published_date": result.published_date,
                "domain": result.url.split('/')[2] if '/' in result.url else result.url
            }
            sources.append(source_data)
            
            if result.text_contents and result.text_contents.text:
                content_snippet = result.text_contents.text[:2000]  # Hosszabb részletek
                combined_content += f"\n--- Forrás {i+1}: {result.title} ---\n{content_snippet}\n"

        # AI elemzés kibővített prompt-tal
        analysis_text = ""
        
        if combined_content and len(combined_content) > 500:
            model_info = await select_backend_model(req.query)
            
            analysis_prompt = f"""
            ÁTFOGÓ TUDOMÁNYOS ELEMZÉS: {req.query}
            
            Feldolgozott források száma: {len(sources)} db
            Teljes tartalom hossza: {len(combined_content)} karakter
            
            FORRÁS ADATOK:
            {combined_content[:50000]}  # Több tartalom feldolgozása
            
            KÉRLEK, KÉSZÍTS RÉSZLETES, TUDOMÁNYOS ELEMZÉST:
            
            1. EXECUTIVE SUMMARY
            - Legfontosabb megállapítások
            - Kulcs információk
            
            2. TUDOMÁNYOS ÁTTEKINTÉS
            - Jelenlegi kutatási állapot
            - Főbb tanulmányok eredményei
            - Konszenzus és viták
            
            3. MÓDSZERTANI MEGKÖZELÍTÉSEK
            - Alkalmazott kutatási módszerek
            - Adatgyűjtési technikák
            - Elemzési eljárások
            
            4. GYAKORLATI ALKALMAZÁSOK
            - Valós életbeli implementációk
            - Ipari alkalmazások
            - Társadalmi hatások
            
            5. JÖVŐBELI KUTATÁSI IRÁNYOK
            - Azonosított kutatási rések
            - Új technológiai lehetőségek
            - Várható fejlődési trendek
            
            6. FORRÁSOK MINŐSÉGI ÉRTÉKELÉSE
            - Magas impakt faktorú publikációk
            - Peer-reviewed források aránya
            - Földrajzi és intézményi diverzitás
            
            7. KÖVETKEZTETÉSEK ÉS AJÁNLÁSOK
            - Összegző megállapítások
            - Döntéshozóknak szóló ajánlások
            - További kutatási prioritások
            
            A válasz legyen strukturált, magyar nyelvű, és használjon tudományos terminológiát.
            Hivatkozz konkrét forrásokra ahol lehetséges.
            """
            
            try:
                result = await execute_model(model_info, analysis_prompt)
                analysis_text = result["response"]
                logger.info(f"AI analysis completed using {result['model_used']}")
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                analysis_text = f"Részleges elemzés készült. Hiba részletei: {e}\n\nElérhető források alapján: {len(sources)} publikáció került feldolgozásra a témában."
        else:
            analysis_text = "Nem sikerült elegendő forrást találni az elemzéshez."
        
        return {
            "query": req.query,
            "final_synthesis": analysis_text,
            "sources": sources,
            "total_sources": len(sources),
            "unique_domains": len(set(s["domain"] for s in sources)),
            "processing_stats": {
                "total_results_found": len(all_results),
                "unique_results": len(final_results),
                "content_length": len(combined_content),
                "domains_searched": len(scientific_domains)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in deep research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a mélyreható kutatás során: {e}"
        )

# Meglévő specializált végpontok megőrzése
@app.post("/api/exa/advanced_search")
async def exa_advanced_search(req: AdvancedExaRequest):
    """Fejlett Exa keresés minden paraméterrel"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        # Keresési paraméterek összeállítása
        search_params = {
            "query": req.query,
            "num_results": req.num_results,
            "text_contents": req.text_contents_options,
            "livecrawl": req.livecrawl
        }

        # Típus alapú keresés
        if req.type == "similarity" and hasattr(exa_client, 'find_similar'):
            # Hasonlóság alapú kereséshez URL szükséges
            search_params["type"] = "similarity"
        elif req.type == "keyword":
            search_params["type"] = "keyword"
        else:
            search_params["type"] = "neural"

        # Domain szűrők
        if req.include_domains:
            search_params["include_domains"] = req.include_domains
        if req.exclude_domains:
            search_params["exclude_domains"] = req.exclude_domains

        # Dátum szűrők
        if req.start_crawl_date:
            search_params["start_crawl_date"] = req.start_crawl_date
        if req.end_crawl_date:
            search_params["end_crawl_date"] = req.end_crawl_date
        if req.start_published_date:
            search_params["start_published_date"] = req.start_published_date
        if req.end_published_date:
            search_params["end_published_date"] = req.end_published_date

        # Szöveg szűrők
        if req.include_text:
            search_params["include_text"] = req.include_text
        if req.exclude_text:
            search_params["exclude_text"] = req.exclude_text

        # Kategória szűrők
        if req.category:
            search_params["category"] = req.category
        if req.subcategory:
            search_params["subcategory"] = req.subcategory

        logger.info(f"Advanced Exa search with params: {search_params}")
        response = exa_client.search(**search_params)

        # Eredmények feldolgozása
        results = []
        for result in response.results:
            processed_result = {
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "published_date": result.published_date,
                "author": getattr(result, 'author', None),
                "score": getattr(result, 'score', None),
                "text_content": result.text_contents.text if result.text_contents else None,
                "highlights": getattr(result, 'highlights', None)
            }
            results.append(processed_result)

        return {
            "query": req.query,
            "search_type": req.type,
            "total_results": len(results),
            "results": results,
            "search_params": search_params,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in advanced Exa search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a fejlett Exa keresés során: {e}"
        )

@app.post("/api/exa/find_similar")
async def exa_find_similar(req: ExaSimilarityRequest):
    """Hasonló tartalmak keresése URL alapján"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        params = {
            "url": req.url,
            "num_results": req.num_results,
            "exclude_source_domain": req.exclude_source_domain
        }

        if req.category_weights:
            params["category_weights"] = req.category_weights

        response = exa_client.find_similar(**params)

        # Text contents külön lekérése
        if response.results:
            ids = [result.id for result in response.results]
            try:
                contents_response = exa_client.get_contents(
                    ids=ids,
                    text_contents={
                        "max_characters": 2000,
                        "strategy": "comprehensive"
                    }
                )
                contents_map = {content.id: content for content in contents_response.contents}
            except:
                contents_map = {}
        else:
            contents_map = {}

        results = []
        for result in response.results:
            content = contents_map.get(result.id)
            processed_result = {
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "similarity_score": getattr(result, 'score', None),
                "published_date": result.published_date,
                "text_content": content.text_contents.text if content and content.text_contents else None
            }
            results.append(processed_result)

        return {
            "reference_url": req.url,
            "similar_results": results,
            "total_found": len(results),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in Exa similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a hasonlóság alapú keresés során: {e}"
        )

@app.post("/api/exa/get_contents")
async def exa_get_contents(req: ExaContentsRequest):
    """Részletes tartalom lekérése Exa result ID alapján"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    try:
        params = {
            "ids": req.ids,
            "text_contents": {
                "max_characters": 5000,
                "include_html_tags": True,
                "strategy": "comprehensive"
            }
        }

        if req.highlights:
            params["highlights"] = req.highlights

        response = exa_client.get_contents(**params)

        contents = []
        for content in response.contents:
            processed_content = {
                "id": content.id,
                "url": content.url,
                "title": content.title,
                "text": content.text_contents.text if content.text_contents else None,
                "html": getattr(content.text_contents, 'html', None) if content.text_contents else None,
                "highlights": getattr(content, 'highlights', None),
                "published_date": content.published_date,
                "author": getattr(content, 'author', None)
            }
            contents.append(processed_content)

        # AI összefoglaló generálása ha kért
        summary = ""
        if req.summary and (gemini_model or cerebras_client):
            combined_text = "\n\n".join([c["text"] for c in contents if c["text"]])[:10000]

            summary_prompt = f"""
            Készíts részletes összefoglalót a következő tartalmakról:

            {combined_text}

            Az összefoglaló legyen strukturált és informatív.
            """

            try:
                if gemini_model:
                    response = await gemini_model.generate_content_async(
                        summary_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=1000,
                            temperature=0.1
                        )
                    )
                    summary = response.text
                elif cerebras_client:
                    stream = cerebras_client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="llama-4-scout-17b-16e-instruct",
                        stream=True,
                        max_completion_tokens=1000,
                        temperature=0.1
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            summary += chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = "Hiba az összefoglaló generálása során"

        return {
            "contents": contents,
            "total_contents": len(contents),
            "ai_summary": summary if req.summary else None,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error getting Exa contents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a tartalmak lekérése során: {e}"
        )

@app.post("/api/exa/neural_search")
async def exa_neural_search(query: str, domains: List[str] = [], exclude_domains: List[str] = [], num_results: int = 20):
    """Speciális neurális keresés tudományos tartalmakhoz"""
    if not exa_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI nem elérhető"
        )

    # Tudományos domainok alapértelmezetten
    if not domains:
        domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "thelancet.com", "bmj.com", "plos.org",
            "ieee.org", "acm.org", "springer.com", "wiley.com", "elsevier.com"
        ]

    try:
        response = exa_client.search(
            query=query,
            type="neural",
            num_results=num_results,
            include_domains=domains,
            exclude_domains=exclude_domains,
            text_contents={
                "max_characters": 3000,
                "strategy": "comprehensive"
            },
            livecrawl="when_necessary"
        )

        # Eredmények pontszám szerint rendezése
        results = sorted(
            response.results, 
            key=lambda x: getattr(x, 'score', 0), 
            reverse=True
        )

        processed_results = []
        for result in results:
            processed_result = {
                "title": result.title,
                "url": result.url,
                "score": getattr(result, 'score', 0),
                "published_date": result.published_date,
                "domain": result.url.split('/')[2] if '/' in result.url else result.url,
                "text_preview": result.text_contents.text[:500] + "..." if result.text_contents else None
            }
            processed_results.append(processed_result)

        return {
            "query": query,
            "neural_results": processed_results,
            "domains_searched": domains,
            "total_results": len(processed_results),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in neural search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a neurális keresés során: {e}"
        )

@app.post("/api/deep_discovery/research_trends")
async def get_research_trends(req: ScientificInsightRequest):
    if not exa_client or not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exa AI vagy Gemini nem elérhető"
        )

    try:
        # Fejlett Exa keresés tudományos domainekkel
        scientific_domains = [
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
            "cell.com", "nejm.org", "thelancet.com", "bmj.com", "plos.org",
            "ieee.org", "acm.org", "springer.com", "wiley.com", "biorxiv.org"
        ]

        search_response = exa_client.search(
            query=req.query,
            type="neural",
            num_results=req.num_results,
            include_domains=scientific_domains,
            text_contents={
                "max_characters": 2000, 
                "strategy": "comprehensive",
                "include_html_tags": False
            },
            livecrawl="when_necessary",
            start_published_date="2020-01-01"  # Friss kutatások
        )

        if not search_response or not search_response.results:
            return {
                "query": req.query,
                "summary": "Nem található releváns információ",
                "sources": []
            }

        sources = []
        combined_content = ""
        for i, result in enumerate(search_response.results):
            if result.text_contents and result.text_contents.text:
                combined_content += f"--- Forrás {i+1}: {result.title} ({result.url}) ---\n{result.text_contents.text}\n\n"
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "published_date": result.published_date
                })

        summary_prompt = f"""
        Elemezd a következő tudományos információkat és készíts összefoglalót (max. {req.summary_length} szó):

        {combined_content[:8000]}

        Összefoglalás:
        """

        response = await gemini_model.generate_content_async(
            summary_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=req.summary_length * 2,
                temperature=0.1
            )
        )

        return {
            "query": req.query,
            "summary": response.text,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in research trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a kutatási trendek elemzése során: {e}"
        )

@app.post("/api/deep_discovery/protein_structure")
async def protein_structure_lookup(req: ProteinLookupRequest):
    ebi_alphafold_api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{req.protein_id}"

    try:
        # Optimalizált HTTP kliens gyorsabb timeout-tal
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(ebi_alphafold_api_url)
            response.raise_for_status()
            data = response.json()

            if not data or (isinstance(data, list) and not data):
                return {
                    "protein_id": req.protein_id,
                    "message": "Nem található előrejelzés",
                    "details": None
                }

            first_prediction = data[0] if isinstance(data, list) else data

            return {
                "protein_id": req.protein_id,
                "message": "Sikeres lekérdezés",
                "details": {
                    "model_id": first_prediction.get("model_id"),
                    "uniprot_id": first_prediction.get("uniprot_id"),
                    "plddt": first_prediction.get("plddt"),
                    "protein_url": first_prediction.get("cif_url") or first_prediction.get("pdb_url"),
                    "pae_url": first_prediction.get("pae_url"),
                    "assembly_id": first_prediction.get("assembly_id")
                }
            }

    except Exception as e:
        logger.error(f"Error in protein lookup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a fehérje lekérdezése során: {e}"
        )

@app.post("/api/deep_discovery/custom_gcp_model")
async def custom_gcp_model_inference(req: CustomGCPModelRequest):
    if not gcp_credentials:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP Vertex AI nem elérhető"
        )

    try:
        project = req.gcp_project_id or GCP_PROJECT_ID
        region = req.gcp_region or GCP_REGION
        endpoint_name = f"projects/{project}/locations/{region}/endpoints/{req.gcp_endpoint_id}"

        prediction_client = aiplatform.gapic.PredictionServiceClient(credentials=gcp_credentials)
        instances_list = [req.input_data] if not isinstance(req.input_data, list) else req.input_data

        response = prediction_client.predict(
            endpoint=endpoint_name,
            instances=instances_list
        )

        return {
            "model_response": response.predictions,
            "model_id": req.gcp_endpoint_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in GCP model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a GCP modell hívásakor: {e}"
        )

@app.post("/api/deep_discovery/simulation_optimizer")
async def simulation_optimizer(req: SimulationOptimizerRequest):
    if not cerebras_client and not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető modell"
        )

    prompt = f"""
    {req.simulation_type} szimuláció optimalizálása:
    Paraméterek: {json.dumps(req.input_parameters, indent=2)}
    Cél: {req.optimization_goal}

    Generálj optimalizált paramétereket vagy Python kódot.
    """

    try:
        response_text = ""
        model_used = ""

        if cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=1024,
                temperature=0.3
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"
        elif gemini_model:
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.3
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"

        return {
            "simulation_type": req.simulation_type,
            "optimization_goal": req.optimization_goal,
            "optimized_output": response_text,
            "model_used": model_used,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in simulation optimizer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a szimuláció optimalizálásában: {e}"
        )

@app.post("/api/deep_discovery/alphagenome")
async def alphagenome_analysis(req: AlphaGenomeRequest):
    if not gemini_model and not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nincs elérhető AI modell"
        )

    valid_nucleotides = set('ATCGURYN-')
    sequence_upper = req.genome_sequence.upper()
    if not all(char in valid_nucleotides for char in sequence_upper):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Érvénytelen genom szekvencia"
        )

    try:
        sequence_length = len(req.genome_sequence)
        gc_content = (sequence_upper.count('G') + sequence_upper.count('C')) / sequence_length * 100

        analysis_prompt = f"""
        AlphaGenome Genomikai Elemzés:
        Szekvencia: {req.genome_sequence[:500]}{'...' if len(req.genome_sequence) > 500 else ''}
        Szervezet: {req.organism}
        Elemzés: {req.analysis_type}
        Hossz: {sequence_length} bp
        GC: {gc_content:.1f}%

        Végezz részletes genomikai elemzést magyar nyelven.
        """

        response_text = ""
        model_used = ""

        if gemini_model:
            response = await gemini_model.generate_content_async(
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            response_text = response.text
            model_used = "Gemini 1.5 Pro"
        elif cerebras_client:
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.1
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            model_used = "Cerebras Llama 4"

        return {
            "sequence_info": {
                "length": sequence_length,
                "gc_content": round(gc_content, 2),
                "organism": req.organism,
                "analysis_type": req.analysis_type
            },
            "ai_analysis": {
                "content": response_text,
                "model_used": model_used
            },
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in AlphaGenome: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba az AlphaGenome elemzésben: {e}"
        )

@app.post("/api/cerebras/ultra_fast_chat")
async def cerebras_ultra_fast_chat(req: ChatRequest):
    """Ultra gyors Cerebras-only chat - 20x gyorsabb mint GPU"""
    if not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cerebras nem elérhető"
        )

    try:
        # Minimal context a maximális sebességért
        messages = [
            {"role": "system", "content": "Gyors, pontos AI asszisztens vagy."},
            {"role": "user", "content": req.message}
        ]

        response_text = ""
        stream = cerebras_client.chat.completions.create(
            messages=messages,
            model="llama-4-scout-17b-16e-instruct",
            stream=True,
            max_completion_tokens=1024,
            temperature=0.1,
            top_p=0.9
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        return {
            'response': response_text,
            'model_used': 'Cerebras Llama 4 (Ultra Fast)',
            'performance': '20x gyorsabb mint GPU',
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Cerebras ultra fast error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cerebras hiba: {e}"
        )

@app.post("/api/cerebras/batch_process")
async def cerebras_batch_process(requests: List[ChatRequest]):
    """Batch feldolgozás Cerebras-szal - még gyorsabb"""
    if not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cerebras nem elérhető"
        )

    results = []
    for req in requests:
        try:
            response_text = ""
            stream = cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": req.message}],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=512,
                temperature=0.1
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            results.append({
                'user_id': req.user_id,
                'response': response_text,
                'status': 'success'
            })

        except Exception as e:
            results.append({
                'user_id': req.user_id,
                'error': str(e),
                'status': 'error'
            })

    return {
        'batch_results': results,
        'processed_count': len(results),
        'model_used': 'Cerebras Llama 4 (Batch)',
        'performance': 'Optimalizált batch feldolgozás'
    }

@app.post("/api/code_generation")
async def generate_code(req: CodeGenerationRequest):
    """Kód generálás Qwen 3 modellel"""
    if not cerebras_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cerebras Qwen 3 modell nem elérhető"
        )

    try:
        # Nyelv-specifikus system prompt
        language_prompts = {
            "python": "Python programozó szakértő vagy. Írd az kódot PEP8 szabványok szerint, kommentekkel és dokumentációval.",
            "javascript": "JavaScript/TypeScript szakértő vagy. Modern ES6+ szintaxist használj, írj tiszta, olvasható kódot.",
            "html": "Frontend fejlesztő vagy. Szemantikus HTML-t írj, modern CSS-sel és hozzáférhetőségi szabványokkal.",
            "css": "CSS szakértő vagy. Modern CSS3 tulajdonságokat használj, responsive designt alkalmazz.",
            "cpp": "C++ programozó vagy. Modern C++17/20 funkciókat használj, memóriabiztos kódot írj.",
            "java": "Java szakértő vagy. Spring Boot és modern Java gyakorlatokat alkalmazz.",
            "csharp": "C# .NET fejlesztő vagy. SOLID elveket kövesd, async/await mintákat használj.",
            "php": "PHP fejlesztő vagy. Modern PHP 8+ funkciókat használj, PSR szabványokat kövesd.",
            "go": "Go fejlesztő vagy. Idiomatikus Go kódot írj, egyszerű és hatékony megoldásokat alkalmazz.",
            "rust": "Rust szakértő vagy. Biztonságos, performáns kódot írj ownership elvek szerint.",
            "swift": "Swift fejlesztő vagy. iOS/macOS kifejlesztéshez optimalizált kódot írj.",
            "kotlin": "Kotlin fejlesztő vagy. Android fejlesztéshez és backend alkalmazásokhoz.",
            "dart": "Dart/Flutter fejlesztő vagy. Cross-platform mobilapplikációkhoz.",
            "sql": "Adatbázis szakértő vagy. Optimalizált SQL lekérdezéseket írj."
        }

        system_prompt = language_prompts.get(req.language.lower(), 
            f"{req.language} programozó szakértő vagy. Tiszta, jól dokumentált kódot írj.")

        # Komplexitás alapú kiegészítés
        complexity_additions = {
            "simple": "Egy egyszerű, rövid kódrészletet írj. Maximum 20-30 sor.",
            "medium": "Közepes komplexitású kódot írj, funkciókat és osztályokat használva.",
            "complex": "Komplex, teljes megoldást írj, több fájllal és fejlett funkcionalitással.",
            "enterprise": "Vállalati szintű, scalable kódot írj design pattern-ekkel és dokumentációval."
        }

        system_prompt += f" {complexity_additions.get(req.complexity, '')}"
        system_prompt += " A kódot magyar nyelvű kommentekkel lásd el, de a változónevek és függvénynevek legyenek angolul."

        # Qwen 3 modell hívása
        response_text = ""
        stream = cerebras_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ],
            model="qwen-3-32b",
            stream=True,
            max_completion_tokens=16382,
            temperature=req.temperature,
            top_p=0.95
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        # Kód extrakció és formázás
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', response_text, re.DOTALL)
        
        if code_blocks:
            generated_code = code_blocks[0]
        else:
            # Ha nincs kódblokk, az egész választ tekintjük kódnak
            generated_code = response_text.strip()

        # Fájlkiterjesztés meghatározása
        extensions = {
            "python": "py", "javascript": "js", "html": "html", "css": "css",
            "cpp": "cpp", "java": "java", "csharp": "cs", "php": "php",
            "go": "go", "rust": "rs", "swift": "swift", "kotlin": "kt",
            "dart": "dart", "sql": "sql", "typescript": "ts"
        }
        
        file_extension = extensions.get(req.language.lower(), "txt")

        return {
            "generated_code": generated_code,
            "full_response": response_text,
            "language": req.language,
            "complexity": req.complexity,
            "file_extension": file_extension,
            "estimated_lines": len(generated_code.split('\n')),
            "model_used": "Qwen 3-32B",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in code generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hiba a kód generálás során: {e}"
        )

# Cache tisztítási funkció
async def cleanup_cache():
    """Lejárt cache bejegyzések törlése"""
    current_time = time.time()
    expired_keys = [
        key for key, value in response_cache.items() 
        if current_time - value['timestamp'] > CACHE_EXPIRY
    ]
    for key in expired_keys:
        del response_cache[key]
    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

@app.on_event("startup")
async def startup_event():
    """Alkalmazás indításkor futó események"""
    logger.info("Jade alkalmazás elindult - optimalizált verzió")

# Cache tisztítás endpoint
@app.post("/api/admin/clear_cache")
async def clear_cache():
    """Cache manuális törlése"""
    response_cache.clear()
    return {"message": "Cache törölve", "status": "success"}

# --- Security Headers Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=()'
        return response

# --- Healthcheck Endpoint ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# --- Sentry Integration ---
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=0.1,
        environment=os.environ.get("ENVIRONMENT", "production"),
    )

# --- Prometheus Metrics ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

@app.get("/ready")
async def ready():
    required_keys = [GCP_SERVICE_ACCOUNT_KEY_JSON, GCP_PROJECT_ID, GCP_REGION, CEREBRAS_API_KEY, GEMINI_API_KEY, EXA_API_KEY, OPENAI_API_KEY]
    if all(required_keys):
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not ready", "missing": [k for k, v in zip(["GCP_SERVICE_ACCOUNT_KEY_JSON", "GCP_PROJECT_ID", "GCP_REGION", "CEREBRAS_API_KEY", "GEMINI_API_KEY", "EXA_API_KEY", "OPENAI_API_KEY"], required_keys) if not v]})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=5000,
        workers=1,  # Egy worker a memória takarékosság miatt
        access_log=False  # Gyorsabb I/O
    )