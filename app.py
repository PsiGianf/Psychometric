import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger
import json
import uuid
import asyncio
import aiofiles
from typing import TypeAlias, TypedDict, Protocol, List, Optional
from pydantic import BaseModel, Field
from functools import lru_cache
import os
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from itertools import groupby

# === Configuración Básica ===
DIRECTORIO_CONVERSACIONES = Path("conversaciones")
DIRECTORIO_CONVERSACIONES.mkdir(exist_ok=True)
logger.add("asistente_psicometrico.log", rotation="500 MB", retention="10 days")

# === Tipos y Modelos ===
class Mensaje(TypedDict):
    rol: str
    contenido: str
    marca_tiempo: str

IdConversacion: TypeAlias = str
ListaMensajes: TypeAlias = List[Mensaje]

class ItemPsicometrico(BaseModel):
    numero: int
    pregunta: str
    categorias_respuesta: dict[str, int]
   
    class Config:
        validate_assignment = True

class EscalaPsicometrica(BaseModel):
    nombre: str
    descripcion: str
    items: List[ItemPsicometrico]
    opciones_respuesta: List[str]
    marco_temporal: str
    version: str = "1.0"
    fecha_creacion: datetime = Field(default_factory=datetime.now)

    def formatear_contexto(self) -> str:
        return f"""
        Escala: {self.nombre} (v{self.version})
        Marco temporal: {self.marco_temporal}
       
        Preguntas y opciones de respuesta:
        {chr(10).join(f"{item.numero}. {item.pregunta}" for item in self.items)}
       
        Opciones de respuesta:
        {chr(10).join(f"{i}: {opcion}" for i, opcion in enumerate(self.opciones_respuesta))}
        """

# === Prompt del Sistema ===
PROMPT_BASE_SISTEMA = """Soy un asistente especializado en psicometría, experto en generar datos sintéticos basados en perfiles psicológicos reales. Mi función es:

1. Analizar la estructura de escalas psicológicas
2. Generar datos sintéticos basados en perfiles psicológicos típicos:
   - Perfil Alto: Tendencia a puntuaciones elevadas (percentil 75-100)
   - Perfil Medio: Puntuaciones moderadas con variabilidad natural (percentil 25-75)
   - Perfil Bajo: Tendencia a puntuaciones reducidas (percentil 0-25)

Para generar datos, necesito conocer:
- Estructura de la escala cargada
- Cantidad de casos a generar
- Distribución deseada de perfiles

¿Qué escala deseas analizar hoy?"""

PATRONES_PSICOMETRICOS = {
    "perfiles": [
        {
            "tipo": "Alto",
            "patron_respuesta": "Consistentemente elevado con ligera variabilidad",
            "rango_percentil": "75-100"
        },
        {
            "tipo": "Medio",
            "patron_respuesta": "Variabilidad natural alrededor de la media",
            "rango_percentil": "25-75"
        },
        {
            "tipo": "Bajo",
            "patron_respuesta": "Consistentemente reducido con ligera variabilidad",
            "rango_percentil": "0-25"
        }
    ],
    "parametros_generacion": {
        "distribucion_perfiles": "30-40-30",
        "variabilidad_intra_perfil": "0.5-1.0 DE"
    }
}

# === Procesador Psicométrico ===
class ProcesadorPsicometrico:
    def __init__(self):
        self.escala_actual: Optional[EscalaPsicometrica] = None
   
    async def procesar_archivo_escala(self, archivo) -> bool:
        try:
            df = pd.read_csv(archivo)
            if len(df.columns) < 3:
                raise ValueError("El archivo debe tener al menos 3 columnas")

            items = [
                ItemPsicometrico(
                    numero=int(float(fila[0])),
                    pregunta=str(fila[1]).strip(),
                    categorias_respuesta={
                        str(df.iloc[0, i+2]): float(fila[df.columns[i+2]])
                        for i in range(len(df.columns[2:]))
                        if pd.notna(fila[df.columns[i+2]])
                    }
                )
                for _, fila in df.iloc[1:].iterrows()
                if pd.notna(fila[0]) and str(fila[0]).strip()
            ]

            if not items:
                raise ValueError("No se pudieron procesar ítems válidos del archivo")

            self.escala_actual = EscalaPsicometrica(
                nombre=Path(archivo.name).stem.replace("_", " ").title(),
                descripcion=df.iloc[0, 1] if len(df) > 0 else "",
                items=items,
                opciones_respuesta=[str(df.iloc[0, i+2]) for i in range(len(df.columns[2:]))],
                marco_temporal=df.iloc[0, 1] if len(df) > 0 else ""
            )
            return True

        except Exception as e:
            logger.error(f"Error procesando archivo de escala: {e}")
            return False

    async def generar_datos_sinteticos(self, n_casos: int, estructura: dict) -> pd.DataFrame:
        """Genera datos sintéticos según especificaciones"""

        return pd.DataFrame()

# === Gestor de Conversaciones ===
class GestorConversaciones:
    def __init__(self):
        self.directorio_base = DIRECTORIO_CONVERSACIONES

    async def crear_conversacion(self) -> IdConversacion:
        id_conversacion = str(uuid.uuid4())
        directorio_conversacion = self.directorio_base / id_conversacion
        directorio_conversacion.mkdir(exist_ok=True)
       
        metadatos = {
            'fecha_creacion': datetime.now().isoformat(),
            'titulo': f'Análisis {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            'ultima_modificacion': datetime.now().isoformat()
        }
       
        await self._guardar_json(directorio_conversacion / 'metadatos.json', metadatos)
        await self._guardar_json(directorio_conversacion / 'mensajes.json', [])
       
        return id_conversacion

    @staticmethod
    async def _guardar_json(ruta: Path, datos: dict) -> None:
        async with aiofiles.open(ruta, 'w') as f:
            await f.write(json.dumps(datos, ensure_ascii=False))

    @staticmethod
    async def _cargar_json(ruta: Path) -> dict:
        async with aiofiles.open(ruta, 'r') as f:
            return json.loads(await f.read())

    async def obtener_conversaciones(self) -> List[dict]:
        conversaciones = []
        for ruta_conv in self.directorio_base.iterdir():
            if (ruta_metadatos := ruta_conv / 'metadatos.json').exists():
                metadatos = await self._cargar_json(ruta_metadatos)
                metadatos['id'] = ruta_conv.name
                conversaciones.append(metadatos)
       
        return sorted(conversaciones, key=lambda x: x['ultima_modificacion'], reverse=True)

    async def obtener_conversacion(self, id_conversacion: IdConversacion) -> dict:
        dir_conv = self.directorio_base / id_conversacion
        return {
            'metadatos': await self._cargar_json(dir_conv / 'metadatos.json'),
            'mensajes': await self._cargar_json(dir_conv / 'mensajes.json')
        }

    async def guardar_mensaje(self, id_conversacion: IdConversacion, mensaje: Mensaje) -> None:
        dir_conv = self.directorio_base / id_conversacion
        mensajes = await self._cargar_json(dir_conv / 'mensajes.json')
        mensajes.append(mensaje)
        await self._guardar_json(dir_conv / 'mensajes.json', mensajes)
       
        metadatos = await self._cargar_json(dir_conv / 'metadatos.json')
        metadatos['ultima_modificacion'] = datetime.now().isoformat()
        await self._guardar_json(dir_conv / 'metadatos.json', metadatos)

    async def actualizar_titulo_conversacion(self, id_conversacion: IdConversacion, nuevo_titulo: str) -> None:
        ruta_metadatos = self.directorio_base / id_conversacion / 'metadatos.json'
        metadatos = await self._cargar_json(ruta_metadatos)
        metadatos['titulo'] = nuevo_titulo
        metadatos['ultima_modificacion'] = datetime.now().isoformat()
        await self._guardar_json(ruta_metadatos, metadatos)

# === Asistente Principal ===
class AsistentePsicometrico:
    def __init__(self):
        self.modelo = ChatOllama(
            model="llama3.1",
            temperature=0.7,
            max_tokens=200,
            context_window=8192,
            gpu_layers=16,
            num_threads=4,
            num_gpu=1,
            mmap=True
        )
        self.patrones_psicometricos = PATRONES_PSICOMETRICOS
        self.procesador_psicometrico = ProcesadorPsicometrico()
        self.gestor_conversaciones = GestorConversaciones()

    async def procesar_mensaje(self, entrada_usuario: str, contexto: ListaMensajes) -> str:
        try:
            mensajes = [SystemMessage(content=self._construir_contexto_sistema())]
           
            # Convertir historial de mensajes (últimos 10)
            for msg in contexto[-10:]:
                clase_mensaje = HumanMessage if msg["rol"] == "usuario" else AIMessage
                mensajes.append(clase_mensaje(content=msg["contenido"]))
           
            mensajes.append(HumanMessage(content=entrada_usuario))
            respuesta = await asyncio.to_thread(self.modelo.invoke, mensajes)
            return respuesta.content

        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            return "Lo siento, tuve un problema al procesar tu mensaje. ¿Podrías reformularlo?"

    def _construir_contexto_sistema(self) -> str:
        contexto = [
            PROMPT_BASE_SISTEMA,
            "\nPATRONES PSICOMÉTRICOS CONOCIDOS:",
            json.dumps(self.patrones_psicometricos, indent=2, ensure_ascii=False)
        ]
       
        if self.procesador_psicometrico.escala_actual:
            contexto.extend([
                "\nESTRUCTURA DE LA ESCALA ACTUAL:",
                self.procesador_psicometrico.escala_actual.formatear_contexto()
            ])
        else:
            contexto.append("\nNo hay escala cargada actualmente.")
       
        return "\n".join(contexto)

# === Continuación de Interfaz de Usuario ===
@st.cache_data(ttl=600)
def obtener_grupo_fecha(fecha_str: str) -> str:
    fecha = datetime.fromisoformat(fecha_str)
    diferencia = datetime.now() - fecha
   
    match diferencia.days:
        case 0: return "Hoy"
        case 1: return "Ayer"
        case d if d < 7: return "Esta semana"
        case d if d < 30: return "Este mes"
        case _: return fecha.strftime("%B %Y")

async def renderizar_barra_lateral(asistente: AsistentePsicometrico):
    with st.sidebar:
        st.title("📊 Análisis Psicométricos")
        termino_busqueda = st.text_input("🔍 Buscar análisis", "")
       
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📝 Nuevo Análisis", type="primary", use_container_width=True):
                id_conv = await asistente.gestor_conversaciones.crear_conversacion()
                st.session_state.conversacion_actual = id_conv
                st.session_state.mensajes = []
                st.rerun()

        st.divider()
        conversaciones = await asistente.gestor_conversaciones.obtener_conversaciones()
       
        if termino_busqueda:
            conversaciones = [
                conv for conv in conversaciones
                if termino_busqueda.lower() in conv['titulo'].lower()
            ]

        if not conversaciones:
            st.info("No se encontraron análisis previos")
            return

        # Agrupar y mostrar conversaciones
        for nombre_grupo, grupo_convs in groupby(
            conversaciones,
            key=lambda x: obtener_grupo_fecha(x['ultima_modificacion'])
        ):
            st.subheader(nombre_grupo)
            grupo_convs = list(grupo_convs)
           
            for conv in grupo_convs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(conv['titulo'], key=f"conv_{conv['id']}", use_container_width=True):
                        datos_conversacion = await asistente.gestor_conversaciones.obtener_conversacion(conv['id'])
                        st.session_state.conversacion_actual = conv['id']
                        st.session_state.mensajes = datos_conversacion['mensajes']
                        st.rerun()
               
                with col2:
                    if st.button("✏️", key=f"edit_{conv['id']}", help="Editar título"):
                        st.session_state.editando_conversacion = conv['id']
                        st.session_state.editando_titulo = conv['titulo']

        if 'editando_conversacion' in st.session_state:
            with st.form(key="formulario_edicion_titulo"):
                nuevo_titulo = st.text_input("Nuevo título", value=st.session_state.editando_titulo)
                if st.form_submit_button("Guardar"):
                    await asistente.gestor_conversaciones.actualizar_titulo_conversacion(
                        st.session_state.editando_conversacion,
                        nuevo_titulo
                    )
                    del st.session_state.editando_conversacion
                    del st.session_state.editando_titulo
                    st.rerun()

async def renderizar_interfaz_chat(asistente: AsistentePsicometrico):
    contenedor_chat = st.container()
   
    with contenedor_chat:
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["rol"]):
                st.markdown(mensaje["contenido"])
                if "marca_tiempo" in mensaje:
                    st.caption(
                        datetime.fromisoformat(mensaje["marca_tiempo"]).strftime("%H:%M")
                    )

    if prompt := st.chat_input("Describe el análisis psicométrico que necesitas...", key="entrada_chat"):
        mensaje_usuario = {
            "rol": "usuario",
            "contenido": prompt,
            "marca_tiempo": datetime.now().isoformat()
        }
       
        st.session_state.mensajes.append(mensaje_usuario)
        with st.chat_message("usuario"):
            st.markdown(prompt)
       
        with st.chat_message("asistente"):
            with st.spinner("Analizando..."):
                respuesta = await asistente.procesar_mensaje(
                    prompt,
                    st.session_state.mensajes
                )
       
        mensaje_asistente = {
            "rol": "asistente",
            "contenido": respuesta,
            "marca_tiempo": datetime.now().isoformat()
        }
       
        st.session_state.mensajes.append(mensaje_asistente)
        with st.chat_message("asistente"):
            st.markdown(respuesta)
       
        if 'conversacion_actual' in st.session_state:
            await asistente.gestor_conversaciones.guardar_mensaje(
                st.session_state.conversacion_actual,
                mensaje_usuario
            )
            await asistente.gestor_conversaciones.guardar_mensaje(
                st.session_state.conversacion_actual,
                mensaje_asistente
            )

async def renderizar_panel_configuracion(asistente: AsistentePsicometrico):
    with st.expander("⚙️ Configuración del Análisis", expanded=False):
        st.subheader("Cargar Escala Psicométrica")
        archivo_escala = st.file_uploader(
            "Cargar escala (CSV)",
            type=['csv'],
            help="Archivo CSV con la estructura de la escala psicométrica"
        )
       
        if archivo_escala:
            if await asistente.procesador_psicometrico.procesar_archivo_escala(archivo_escala):
                st.success("✅ Escala cargada correctamente")
                st.subheader("📋 Estructura de la Escala")
                st.markdown(asistente.procesador_psicometrico.escala_actual.formatear_contexto())
async def main():
    st.set_page_config(
        page_title="📊 Asistente Psicométrico",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    # Inicializar estados de sesión
    if 'asistente' not in st.session_state:
        st.session_state.asistente = AsistentePsicometrico()
    if 'mensajes' not in st.session_state:
        st.session_state.mensajes = []
    if 'conversacion_actual' not in st.session_state:
        conversaciones = await st.session_state.asistente.gestor_conversaciones.obtener_conversaciones()
        if not conversaciones:
            id_conv = await st.session_state.asistente.gestor_conversaciones.crear_conversacion()
            st.session_state.conversacion_actual = id_conv
        else:
            st.session_state.conversacion_actual = conversaciones[0]['id']
   
    await renderizar_barra_lateral(st.session_state.asistente)
   
    # Contenedor principal
    with st.container():
        col1, col2 = st.columns([7, 3])
       
        with col1:
            st.title("📊 Asistente de Análisis Psicométrico")
            st.markdown(
                "<p style='font-size:18px;'>Análisis factorial y validación de escalas psicométricas "
                "mediante datos sintéticos. Versión 1.0</p>",
                unsafe_allow_html=True
            )
            await renderizar_interfaz_chat(st.session_state.asistente)
       
        with col2:
            await renderizar_panel_configuracion(st.session_state.asistente)
   
    # Pie de página
    st.markdown(
        """
        <hr style='margin-top: 50px;'>
        <p style='font-size:13px; text-align:center;'>
        ⚠️ Este es un asistente automatizado para análisis psicométrico.
        Los resultados deben ser validados por profesionales en psicometría.
        <br><br>
        © 2025 Asistente de Análisis Psicométrico | Versión 1.0
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    asyncio.run(main())
