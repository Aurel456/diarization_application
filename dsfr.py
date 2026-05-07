"""
dsfr_streamlit.py — Applique la charte DSFR (thème clair) à une application Streamlit.
v3.0 - Refonte exhaustive avec mapping composant par composant.

Usage:
    import streamlit as st
    import dsfr_streamlit as dsfr
    dsfr.apply()
"""

from __future__ import annotations
import streamlit as st

__version__ = "0.4.0"
__all__ = ["apply"]

# URL de base pour les assets DSFR (polices, icônes, favicon)
_DSFR_CDN = "https://unpkg.com/@gouvfr/dsfr@1.14.4/dist"

# -----------------------------------------------------------------------------
# Palette DSFR Thème Clair (Basée sur la documentation officielle)
# -----------------------------------------------------------------------------
_PALETTE = {
    # Couleurs fonctionnelles principales
    "blue-france":          "#000091",
    "blue-france-hover":    "#1212ff",
    "blue-france-active":   "#2323ff",
    "blue-france-soft":     "#e3e3fd",

    "red-marianne":         "#e1000f",
    "red-marianne-hover":   "#ff1e1e",
    "red-marianne-active":  "#ff3b3b",
    "red-marianne-soft":    "#ffe9e9",

    # Échelle de gris
    "grey-1000":            "#161616", # Texte principal
    "grey-900":             "#3a3a3a", # Texte secondaire / Bordures actives
    "grey-800":             "#666666", # Texte de mention
    "grey-700":             "#929292", # Texte désactivé
    "grey-600":             "#dddddd", # Bordures par défaut
    "grey-500":             "#e5e5e5", # Fond désactivé
    "grey-400":             "#eeeeee", # Fond input / Fond contrasté
    "grey-300":             "#f6f6f6", # Fond alt
    "grey-200":             "#ffffff", # Fond principal / Fond overlap

    # Couleurs de feedback (Success, Info, Warning, Error)
    "success":              "#18753c",
    "success-bg":           "#b8fec9",
    "info":                 "#0063cb",
    "info-bg":              "#e8edff",
    "warning":              "#b34000",
    "warning-bg":           "#ffe9e6",
    "error":                "#ce0500",
    "error-bg":             "#ffe9e9",

    # Focus
    "focus":                "#0a76f6",
}

def _css_variables() -> str:
    """Génère le bloc de variables CSS :root."""
    return f"""
        :root {{
            /* Mapping sémantique vers les variables natives Streamlit */
            --background-color: {_PALETTE['grey-200']};
            --secondary-background-color: {_PALETTE['grey-300']};
            --text-color: {_PALETTE['grey-1000']};
            --primary-color: {_PALETTE['blue-france']};
            --font: 'Marianne', arial, sans-serif;

            /* Variables DSFR pour usage avancé */
            --dsfr-blue: {_PALETTE['blue-france']};
            --dsfr-blue-hover: {_PALETTE['blue-france-hover']};
            --dsfr-blue-active: {_PALETTE['blue-france-active']};
            --dsfr-blue-soft: {_PALETTE['blue-france-soft']};

            --dsfr-red: {_PALETTE['red-marianne']};
            --dsfr-red-hover: {_PALETTE['red-marianne-hover']};
            --dsfr-red-soft: {_PALETTE['red-marianne-soft']};

            --dsfr-text: {_PALETTE['grey-1000']};
            --dsfr-text-mention: {_PALETTE['grey-800']};
            --dsfr-text-disabled: {_PALETTE['grey-700']};

            --dsfr-bg: {_PALETTE['grey-200']};
            --dsfr-bg-alt: {_PALETTE['grey-300']};
            --dsfr-bg-contrast: {_PALETTE['grey-400']};
            --dsfr-bg-overlap: {_PALETTE['grey-200']};
            --dsfr-bg-disabled: {_PALETTE['grey-500']};

            --dsfr-border: {_PALETTE['grey-600']};
            --dsfr-border-plain: {_PALETTE['grey-900']};

            --dsfr-input-bg: {_PALETTE['grey-400']};
            --dsfr-input-border: {_PALETTE['grey-900']};

            --dsfr-success: {_PALETTE['success']};
            --dsfr-success-bg: {_PALETTE['success-bg']};
            --dsfr-info: {_PALETTE['info']};
            --dsfr-info-bg: {_PALETTE['info-bg']};
            --dsfr-warning: {_PALETTE['warning']};
            --dsfr-warning-bg: {_PALETTE['warning-bg']};
            --dsfr-error: {_PALETTE['error']};
            --dsfr-error-bg: {_PALETTE['error-bg']};

            --dsfr-focus: {_PALETTE['focus']};
        }}
    """

# -----------------------------------------------------------------------------
# CSS DSFR Complet (Police, Typographie, Composants)
# -----------------------------------------------------------------------------

_FONTS_CSS = f"""
/* Police Marianne - DSFR */
@font-face {{
    font-family: 'Marianne';
    src: url('{_DSFR_CDN}/fonts/Marianne-Regular.woff2') format('woff2');
    font-weight: 400; font-style: normal; font-display: swap;
}}
@font-face {{
    font-family: 'Marianne';
    src: url('{_DSFR_CDN}/fonts/Marianne-Medium.woff2') format('woff2');
    font-weight: 500; font-style: normal; font-display: swap;
}}
@font-face {{
    font-family: 'Marianne';
    src: url('{_DSFR_CDN}/fonts/Marianne-Bold.woff2') format('woff2');
    font-weight: 700; font-style: normal; font-display: swap;
}}

/* Police Spectral (pour les titres) - DSFR */
@font-face {{
    font-family: 'Spectral';
    src: url('{_DSFR_CDN}/fonts/Spectral-Regular.woff2') format('woff2');
    font-weight: 400; font-style: normal; font-display: swap;
}}
@font-face {{
    font-family: 'Spectral';
    src: url('{_DSFR_CDN}/fonts/Spectral-Bold.woff2') format('woff2');
    font-weight: 700; font-style: normal; font-display: swap;
}}
"""

_TYPOGRAPHY_CSS = """
/* Typographie globale DSFR */
html, body, .stApp {
    font-family: 'Marianne', arial, sans-serif;
    font-weight: 400;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Spectral', georgia, serif;
    font-weight: 700;
    color: var(--dsfr-text);
    margin-bottom: 1.5rem;
}
h1 { font-size: 2.5rem; line-height: 3rem; }
h2 { font-size: 2rem; line-height: 2.5rem; }
h3 { font-size: 1.75rem; line-height: 2.25rem; }
h4 { font-size: 1.5rem; line-height: 2rem; }
h5 { font-size: 1.25rem; line-height: 1.75rem; }
h6 { font-size: 1.125rem; line-height: 1.5rem; }

p, li, blockquote, figcaption, label, legend, caption, th, td {
    font-family: 'Marianne', arial, sans-serif;
}
p, li { line-height: 1.5rem; color: var(--dsfr-text); }

a, a:visited { color: var(--dsfr-blue); text-decoration: underline; text-underline-offset: 2px; }
a:hover { color: var(--dsfr-blue-hover); }

*:focus-visible {
    outline: 2px solid var(--dsfr-focus);
    outline-offset: 2px;
}
::selection { background-color: var(--dsfr-blue); color: #fff; }
"""

# Mapping exhaustif des composants Streamlit -> DSFR
_COMPONENTS_CSS = """
/* -------------------------------------------------------------------------- */
/* 1. STRUCTURE & LAYOUT                                                      */
/* -------------------------------------------------------------------------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--dsfr-bg);
    color: var(--dsfr-text);
}

/* En-tête Streamlit (optionnel, souvent caché) */
header[data-testid="stHeader"] {
    background-color: var(--dsfr-bg-alt);
    border-bottom: 1px solid var(--dsfr-border);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--dsfr-bg-alt);
    border-right: 1px solid var(--dsfr-border);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stHeading"] {
    color: var(--dsfr-text);
}

/* -------------------------------------------------------------------------- */
/* 2. TYPOGRAPHIE & MARQUAGE                                                  */
/* -------------------------------------------------------------------------- */
[data-testid="stMarkdownContainer"] {
    color: var(--dsfr-text);
}
[data-testid="stCaptionContainer"] {
    color: var(--dsfr-text-mention);
    font-size: 0.875rem;
}

/* -------------------------------------------------------------------------- */
/* 3. BOUTONS (Primary, Secondary, Tertiary, Download, Form)                  */
/* -------------------------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    font-family: 'Marianne', arial, sans-serif;
    font-weight: 500;
    font-size: 1rem;
    min-height: 2.5rem;
    padding: 0.5rem 1rem;
    border-radius: 0; /* DSFR: bordures franches */
    box-shadow: none;
    transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    line-height: 1.5rem;
    border: 1px solid transparent;
}

/* Primaire */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background-color: var(--dsfr-blue);
    border-color: var(--dsfr-blue);
    color: #ffffff;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background-color: var(--dsfr-blue-hover);
    border-color: var(--dsfr-blue-hover);
}
.stButton > button[kind="primary"]:active,
.stDownloadButton > button[kind="primary"]:active,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:active {
    background-color: var(--dsfr-blue-active);
    border-color: var(--dsfr-blue-active);
}

/* Secondaire */
.stButton > button[kind="secondary"],
.stFormSubmitButton > button[kind="secondaryFormSubmit"] {
    background-color: transparent;
    border-color: var(--dsfr-blue);
    color: var(--dsfr-blue);
}
.stButton > button[kind="secondary"]:hover,
.stFormSubmitButton > button[kind="secondaryFormSubmit"]:hover {
    background-color: var(--dsfr-blue-soft);
    color: var(--dsfr-blue);
}

/* Tertiaire */
.stButton > button[kind="tertiary"] {
    background-color: transparent;
    border-color: transparent;
    color: var(--dsfr-blue);
}
.stButton > button[kind="tertiary"]:hover {
    background-color: var(--dsfr-bg-alt);
}

/* Désactivé */
.stButton > button:disabled,
.stDownloadButton > button:disabled,
.stFormSubmitButton > button:disabled {
    background-color: var(--dsfr-bg-disabled);
    border-color: var(--dsfr-bg-disabled);
    color: var(--dsfr-text-disabled);
}

/* -------------------------------------------------------------------------- */
/* 4. CHAMPS DE SAISIE (Text, Number, TextArea, Select, Date, Time)           */
/* -------------------------------------------------------------------------- */
/* Conteneur racine */
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div,
.stDateInput div[data-baseweb="input"] > div,
.stTimeInput div[data-baseweb="input"] > div {
    background-color: var(--dsfr-input-bg);
    border-radius: 0.25rem 0.25rem 0 0;
    border: none;
    border-bottom: 2px solid var(--dsfr-input-border);
    box-shadow: none;
}

/* Champ de saisie lui-même */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: transparent;
    color: var(--dsfr-text);
    caret-color: var(--dsfr-blue);
    font-family: 'Marianne', arial, sans-serif;
}

/* État focus */
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within,
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="textarea"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {
    border-bottom-color: var(--dsfr-blue);
    outline: 2px solid var(--dsfr-focus);
    outline-offset: 2px;
}

/* Labels */
.stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label {
    color: var(--dsfr-text);
    font-weight: 500;
    font-size: 1rem;
}

/* -------------------------------------------------------------------------- */
/* 5. LISTES DÉROULANTES & POPOVERS (Select, Multiselect, DatePicker)         */
/* -------------------------------------------------------------------------- */
div[data-baseweb="popover"] {
    background-color: var(--dsfr-bg-overlap);
    border: 1px solid var(--dsfr-border);
    border-radius: 0;
    box-shadow: 0 2px 6px rgba(0,0,18,0.16);
    padding: 0.25rem 0;
}

div[data-baseweb="popover"] li {
    font-family: 'Marianne', arial, sans-serif;
    padding: 0.5rem 1rem;
    color: var(--dsfr-text);
}
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="popover"] li:hover {
    background-color: var(--dsfr-blue-soft);
    color: var(--dsfr-text);
}

/* Tags (Multiselect) */
div[data-baseweb="tag"] {
    background-color: var(--dsfr-blue-soft);
    color: var(--dsfr-blue);
    border-radius: 0.25rem;
    font-weight: 500;
}

/* -------------------------------------------------------------------------- */
/* 6. SLIDERS (Slider, SelectSlider)                                          */
/* -------------------------------------------------------------------------- */
.stSlider [data-baseweb="slider"] [role="slider"],
.stSelectSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--dsfr-blue);
    box-shadow: 0 0 0 1px var(--dsfr-blue);
}
.stSlider [data-baseweb="slider"] > div > div:first-child,
.stSelectSlider [data-baseweb="slider"] > div > div:first-child {
    background-color: var(--dsfr-blue) !important;
}

/* -------------------------------------------------------------------------- */
/* 7. CHECKBOX, RADIO, TOGGLE                                                 */
/* -------------------------------------------------------------------------- */
.stCheckbox [data-baseweb="checkbox"] span[role="checkbox"][aria-checked="true"],
.stToggle [role="checkbox"][aria-checked="true"] {
    background-color: var(--dsfr-blue);
    border-color: var(--dsfr-blue);
}
.stCheckbox label, .stRadio label, .stToggle label {
    color: var(--dsfr-text);
}
.stRadio [role="radio"][aria-checked="true"] > div:first-child {
    background-color: var(--dsfr-blue);
    border-color: var(--dsfr-blue);
}

/* -------------------------------------------------------------------------- */
/* 8. ONGLETS (Tabs)                                                          */
/* -------------------------------------------------------------------------- */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid var(--dsfr-border);
    gap: 0;
}
div[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--dsfr-text);
    font-weight: 500;
    padding: 0.75rem 1rem;
    border-radius: 0;
    font-family: 'Marianne', arial, sans-serif;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom-color: var(--dsfr-blue);
    color: var(--dsfr-blue);
    font-weight: 700;
}
/* Cacher les surlignages par défaut de Streamlit */
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* -------------------------------------------------------------------------- */
/* 9. ALERTES (Success, Info, Warning, Error)                                 */
/* -------------------------------------------------------------------------- */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
    border-radius: 0;
    border-left: 4px solid var(--dsfr-info);
    background-color: var(--dsfr-info-bg);
    color: var(--dsfr-text);
    padding: 1rem 1.25rem;
    font-family: 'Marianne', arial, sans-serif;
}
[data-testid="stAlertContentSuccess"] {
    border-left-color: var(--dsfr-success);
    background-color: var(--dsfr-success-bg);
}
[data-testid="stAlertContentWarning"] {
    border-left-color: var(--dsfr-warning);
    background-color: var(--dsfr-warning-bg);
}
[data-testid="stAlertContentError"] {
    border-left-color: var(--dsfr-error);
    background-color: var(--dsfr-error-bg);
}
[data-testid="stAlertContentInfo"] {
    border-left-color: var(--dsfr-info);
    background-color: var(--dsfr-info-bg);
}

/* -------------------------------------------------------------------------- */
/* 10. EXPANDER (Accordéon)                                                   */
/* -------------------------------------------------------------------------- */
[data-testid="stExpander"] {
    border: 1px solid var(--dsfr-border);
    border-radius: 0;
    background-color: var(--dsfr-bg);
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > div:first-child {
    background-color: var(--dsfr-bg-alt);
    color: var(--dsfr-blue);
    font-weight: 500;
    font-family: 'Marianne', arial, sans-serif;
    padding: 0.75rem 1rem;
}

/* -------------------------------------------------------------------------- */
/* 11. MODALES & POPOVERS (Dialog, Popover)                                   */
/* -------------------------------------------------------------------------- */
div[role="dialog"] {
    background-color: var(--dsfr-bg-overlap);
    border: 1px solid var(--dsfr-border);
    border-radius: 0;
    color: var(--dsfr-text);
    box-shadow: 0 4px 12px rgba(0,0,18,0.16);
}
[data-testid="stPopoverBody"] {
    background-color: var(--dsfr-bg-overlap);
    border: 1px solid var(--dsfr-border);
    border-radius: 0;
    color: var(--dsfr-text);
}

/* -------------------------------------------------------------------------- */
/* 12. CHAT (Chat Input, Messages)                                            */
/* -------------------------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background-color: transparent;
    border-left: 4px solid var(--dsfr-border);
    border-radius: 0;
    padding-left: 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left-color: var(--dsfr-blue);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left-color: var(--dsfr-red);
}
[data-testid="stBottomBlockContainer"] {
    background-color: var(--dsfr-bg);
    border-top: 1px solid var(--dsfr-border);
}
[data-testid="stChatInput"] > div {
    background-color: var(--dsfr-input-bg);
    border: none;
    border-bottom: 2px solid var(--dsfr-input-border);
    border-radius: 0.25rem 0.25rem 0 0;
}

/* -------------------------------------------------------------------------- */
/* 13. MÉTRIQUES (Metric)                                                     */
/* -------------------------------------------------------------------------- */
[data-testid="stMetric"] {
    background-color: var(--dsfr-bg-alt);
    border-left: 4px solid var(--dsfr-blue);
    padding: 1rem 1.25rem;
    border-radius: 0;
}
[data-testid="stMetricLabel"] {
    color: var(--dsfr-text-mention);
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: var(--dsfr-text);
    font-weight: 700;
    font-size: 2rem;
}

/* -------------------------------------------------------------------------- */
/* 14. PROGRESS, STATUS, TOAST, SPINNER                                      */
/* -------------------------------------------------------------------------- */
.stProgress > div > div > div > div {
    background-color: var(--dsfr-blue);
}
[data-testid="stStatus"], [data-testid="stStatusWidget"] {
    background-color: var(--dsfr-bg-alt);
    border: 1px solid var(--dsfr-border);
    border-radius: 0;
    color: var(--dsfr-text);
}
[data-testid="stToast"] {
    background-color: var(--dsfr-bg-overlap);
    border: 1px solid var(--dsfr-border);
    border-left: 4px solid var(--dsfr-blue);
    border-radius: 0;
    color: var(--dsfr-text);
}
.stSpinner > div {
    border-top-color: var(--dsfr-blue) !important;
    border-left-color: var(--dsfr-blue) !important;
}

/* -------------------------------------------------------------------------- */
/* 15. TABLEAUX (DataFrame, Table)                                            */
/* -------------------------------------------------------------------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--dsfr-border);
    background-color: var(--dsfr-bg);
    border-radius: 0;
}
[data-testid="stTable"] thead th {
    background-color: var(--dsfr-bg-alt);
    color: var(--dsfr-text);
    font-weight: 700;
    border-bottom: 1px solid var(--dsfr-border);
    font-family: 'Marianne', arial, sans-serif;
}
[data-testid="stTable"] tbody td {
    border-bottom: 1px solid var(--dsfr-border);
    color: var(--dsfr-text);
}

/* -------------------------------------------------------------------------- */
/* 16. CODE & JSON                                                            */
/* -------------------------------------------------------------------------- */
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stJson"] pre {
    background-color: var(--dsfr-bg-alt);
    color: var(--dsfr-text);
    border: 1px solid var(--dsfr-border);
    border-radius: 0.25rem;
}

/* -------------------------------------------------------------------------- */
/* 17. UPLOAD DE FICHIERS (File Uploader)                                     */
/* -------------------------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--dsfr-bg-alt);
    border: 1px dashed var(--dsfr-border-plain);
    border-radius: 0;
    color: var(--dsfr-text);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--dsfr-blue);
    background-color: var(--dsfr-blue-soft);
}

/* -------------------------------------------------------------------------- */
/* 18. FORMULAIRES (Form, Columns)                                            */
/* -------------------------------------------------------------------------- */
[data-testid="stForm"] {
    border: 1px solid var(--dsfr-border);
    background-color: var(--dsfr-bg);
    padding: 1.5rem;
    border-radius: 0;
}
[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

/* -------------------------------------------------------------------------- */
/* 19. LIENS & BOUTONS DE COPIE (Link Button, Copy Button)                    */
/* -------------------------------------------------------------------------- */
.stLinkButton > a {
    color: var(--dsfr-blue);
    text-decoration: underline;
    text-underline-offset: 2px;
    font-weight: 500;
}
.stLinkButton > a:hover {
    color: var(--dsfr-blue-hover);
}

/* -------------------------------------------------------------------------- */
/* 20. MENU DE NAVIGATION (Page Link, Navigation)                             */
/* -------------------------------------------------------------------------- */
[data-testid="stPageLink-NavLink"] {
    color: var(--dsfr-text) !important;
    border-radius: 0;
    border-left: 2px solid transparent;
}
[data-testid="stPageLink-NavLink"]:hover {
    background-color: var(--dsfr-bg-alt);
    color: var(--dsfr-blue) !important;
}
[data-testid="stPageLink-NavLink"][aria-current="page"] {
    border-left-color: var(--dsfr-blue);
    background-color: var(--dsfr-blue-soft);
    color: var(--dsfr-blue) !important;
    font-weight: 500;
}

/* -------------------------------------------------------------------------- */
/* 21. BARRE LATÉRALE (Sidebar) - Éléments spécifiques                        */
/* -------------------------------------------------------------------------- */
[data-testid="stSidebarNavLink"] {
    color: var(--dsfr-text) !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background-color: var(--dsfr-bg-contrast);
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: var(--dsfr-blue-soft);
    color: var(--dsfr-blue) !important;
    border-left: 4px solid var(--dsfr-blue);
}
"""

def _build_css() -> str:
    """Assemble tous les blocs CSS."""
    return (
        _css_variables() +
        _FONTS_CSS +
        _TYPOGRAPHY_CSS +
        _COMPONENTS_CSS
    )

def apply() -> None:
    """
    Applique la charte DSFR (thème clair uniquement) à l'application Streamlit courante.
    """
    # Charge les fonts DSFR via un <link> (plus fiable que @font-face seul sous Streamlit)
    st.markdown(
        f'<link rel="preconnect" href="https://unpkg.com" crossorigin>'
        f'<link rel="stylesheet" href="{_DSFR_CDN}/fonts/fonts.min.css">',
        unsafe_allow_html=True,
    )
    css = _build_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)