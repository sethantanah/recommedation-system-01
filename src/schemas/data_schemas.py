from pydantic import BaseModel
class ModelData(BaseModel):
    model_type: str
    framework_library: str
    input_data: str
    institution: str
    action: str
    license: str
    github_stars1000: str
    citations: str
    model_sizemb: str
    memory_requirementtraining: str
