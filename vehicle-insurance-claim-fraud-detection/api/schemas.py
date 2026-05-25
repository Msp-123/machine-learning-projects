from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ClaimRequest(BaseModel):
    Month: Optional[str] = None
    WeekOfMonth: Optional[int] = None
    DayOfWeek: Optional[str] = None
    Make: Optional[str] = None
    AccidentArea: Optional[str] = None
    DayOfWeekClaimed: Optional[str] = None
    MonthClaimed: Optional[str] = None
    WeekOfMonthClaimed: Optional[int] = None
    Sex: Optional[str] = None
    MaritalStatus: Optional[str] = None
    Age: Optional[float] = None
    Fault: Optional[str] = None
    PolicyType: Optional[str] = None
    VehicleCategory: Optional[str] = None
    VehiclePrice: Optional[str] = None
    PolicyNumber: Optional[int] = None
    RepNumber: Optional[int] = None
    Deductible: Optional[int] = None
    DriverRating: Optional[int] = None
    Days_Policy_Accident: Optional[str] = None
    Days_Policy_Claim: Optional[str] = None
    PastNumberOfClaims: Optional[str] = None
    AgeOfVehicle: Optional[str] = None
    AgeOfPolicyHolder: Optional[str] = None
    PoliceReportFiled: Optional[str] = None
    WitnessPresent: Optional[str] = None
    AgentType: Optional[str] = None
    NumberOfSuppliments: Optional[str] = None
    AddressChange_Claim: Optional[str] = None
    NumberOfCars: Optional[str] = None
    Year: Optional[int] = None
    BasePolicy: Optional[str] = None

    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: int
    prediction_label: str
    risk_level: str
    recommendation: str


class BatchPredictionRequest(BaseModel):
    claims: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    threshold: float


class ExplanationReason(BaseModel):
    processed_feature: str
    raw_column: Optional[str] = None
    category_value: Optional[str] = None
    feature_value: float
    shap_value: float
    absolute_shap_value: float
    direction: str
    reason: str


class ExplanationResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: int
    prediction_label: str
    risk_level: str
    recommendation: str
    top_reasons: List[ExplanationReason]