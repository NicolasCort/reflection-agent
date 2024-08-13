from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Importa la clase Reflection
from app import Reflection

# Instancia de FastAPI
app = FastAPI()

# Modelo de datos que se espera como entrada
class ReflectionRequest(BaseModel):
    task: str
    reflection_prompt: str
    generation_prompt: str

# Endpoint para invocar la clase Reflection
@app.post("/reflect/")
def reflect(request: ReflectionRequest):
    try:
        # Crear instancia de Reflection
        reflection_instance = Reflection(
            task=request.task,
            reflection_prompt=request.reflection_prompt,
            generation_prompt=request.generation_prompt
        )

        # Ejecutar el método build_agent para obtener la respuesta
        response = reflection_instance.build_agent()

        # Retornar la respuesta obtenida
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
