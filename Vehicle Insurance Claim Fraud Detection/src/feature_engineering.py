import numpy as np
import pandas as pd


# =========================================================
# Feature engineering registry
# =========================================================

FEATURE_ENGINEERING_STEPS = []


def register_feature_step(name: str, description: str = ""):
    """
    Decorator to register feature engineering functions.

    This makes the feature engineering pipeline modular.
    To add a new feature group, create a new function and decorate it with:

    @register_feature_step(name="step_name", description="...")
    def add_some_features(df):
        ...
        return df
    """

    def decorator(func):
        FEATURE_ENGINEERING_STEPS.append({
            "name": name,
            "description": description,
            "function": func
        })
        return func

    return decorator


# =========================================================
# Helper functions
# =========================================================

def column_exists(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns


def safe_equals(df: pd.DataFrame, column: str, value) -> pd.Series:
    """
    Return binary flag where column equals value.
    If column does not exist, return 0 for all rows.
    """

    if not column_exists(df, column):
        return pd.Series(0, index=df.index)

    return (df[column] == value).astype(int)


def safe_isin(df: pd.DataFrame, column: str, values: list) -> pd.Series:
    """
    Return binary flag where column value is in values.
    If column does not exist, return 0 for all rows.
    """

    if not column_exists(df, column):
        return pd.Series(0, index=df.index)

    return df[column].isin(values).astype(int)


def safe_not_equals(df: pd.DataFrame, column: str, value) -> pd.Series:
    """
    Return binary flag where column does not equal value.
    If column does not exist, return 0 for all rows.
    """

    if not column_exists(df, column):
        return pd.Series(0, index=df.index)

    return (df[column] != value).astype(int)


def safe_numeric_greater_equal(df: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    """
    Return binary flag where numeric column >= threshold.
    If column does not exist, return 0 for all rows.
    """

    if not column_exists(df, column):
        return pd.Series(0, index=df.index)

    values = pd.to_numeric(df[column], errors="coerce")

    return (values >= threshold).astype(int)


def safe_numeric_less_than(df: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    """
    Return binary flag where numeric column < threshold.
    If column does not exist, return 0 for all rows.
    """

    if not column_exists(df, column):
        return pd.Series(0, index=df.index)

    values = pd.to_numeric(df[column], errors="coerce")

    return (values < threshold).astype(int)


# =========================================================
# Step 1: Cleaning-related features
# =========================================================

@register_feature_step(
    name="clean_special_values",
    description="Handles suspicious values such as Age=0 and unknown claimed date values."
)
def clean_special_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if column_exists(df, "Age"):
        df["Age_Zero_Flag"] = np.where(df["Age"] == 0, 1, 0)
        df["Age"] = df["Age"].replace(0, np.nan)

    if column_exists(df, "MonthClaimed"):
        df["MonthClaimed"] = df["MonthClaimed"].astype(str).replace("0", "Unknown")

    if column_exists(df, "DayOfWeekClaimed"):
        df["DayOfWeekClaimed"] = df["DayOfWeekClaimed"].astype(str).replace("0", "Unknown")

    return df


# =========================================================
# Step 2: Fault features
# =========================================================

@register_feature_step(
    name="fault_features",
    description="Creates features related to who was at fault in the accident."
)
def add_fault_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_policy_holder_fault"] = safe_equals(df, "Fault", "Policy Holder")
    df["is_third_party_fault"] = safe_equals(df, "Fault", "Third Party")

    return df


# =========================================================
# Step 3: Police report and witness features
# =========================================================

@register_feature_step(
    name="report_and_witness_features",
    description="Creates police report and witness-related fraud signal features."
)
def add_report_and_witness_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_no_police_report"] = safe_equals(df, "PoliceReportFiled", "No")
    df["is_police_report_filed"] = safe_equals(df, "PoliceReportFiled", "Yes")

    df["is_no_witness"] = safe_equals(df, "WitnessPresent", "No")
    df["is_witness_present"] = safe_equals(df, "WitnessPresent", "Yes")

    return df


# =========================================================
# Step 4: Agent features
# =========================================================

@register_feature_step(
    name="agent_features",
    description="Creates features related to internal or external agents."
)
def add_agent_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_external_agent"] = safe_equals(df, "AgentType", "External")
    df["is_internal_agent"] = safe_equals(df, "AgentType", "Internal")

    return df


# =========================================================
# Step 5: Address change features
# =========================================================

@register_feature_step(
    name="address_change_features",
    description="Creates features related to address changes before claim."
)
def add_address_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_address_change"] = safe_not_equals(
        df,
        "AddressChange_Claim",
        "no change"
    )

    df["is_recent_address_change"] = safe_isin(
        df,
        "AddressChange_Claim",
        ["under 6 months", "1 year"]
    )

    return df


# =========================================================
# Step 6: Policy timing features
# =========================================================

@register_feature_step(
    name="policy_timing_features",
    description="Creates features related to accident and claim timing after policy start."
)
def add_policy_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    early_values = ["1 to 7", "8 to 15", "15 to 30"]

    df["is_early_policy_accident"] = safe_isin(
        df,
        "Days_Policy_Accident",
        early_values
    )

    df["is_policy_accident_none"] = safe_equals(
        df,
        "Days_Policy_Accident",
        "none"
    )

    df["is_early_policy_claim"] = safe_isin(
        df,
        "Days_Policy_Claim",
        early_values
    )

    df["is_policy_claim_none"] = safe_equals(
        df,
        "Days_Policy_Claim",
        "none"
    )

    return df


# =========================================================
# Step 7: Past claim features
# =========================================================

@register_feature_step(
    name="past_claim_features",
    description="Creates features related to previous claim history."
)
def add_past_claim_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_past_claims"] = safe_not_equals(
        df,
        "PastNumberOfClaims",
        "none"
    )

    df["has_multiple_past_claims"] = safe_isin(
        df,
        "PastNumberOfClaims",
        ["2 to 4", "more than 4"]
    )

    df["has_more_than_4_past_claims"] = safe_equals(
        df,
        "PastNumberOfClaims",
        "more than 4"
    )

    return df


# =========================================================
# Step 8: Vehicle features
# =========================================================

@register_feature_step(
    name="vehicle_features",
    description="Creates vehicle category, vehicle age and vehicle price features."
)
def add_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_high_vehicle_price"] = safe_isin(
        df,
        "VehiclePrice",
        ["60000 to 69000", "more than 69000"]
    )

    df["is_low_vehicle_price"] = safe_equals(
        df,
        "VehiclePrice",
        "less than 20000"
    )

    df["is_old_vehicle"] = safe_isin(
        df,
        "AgeOfVehicle",
        ["7 years", "more than 7"]
    )

    df["is_new_vehicle"] = safe_isin(
        df,
        "AgeOfVehicle",
        ["new", "2 years"]
    )

    df["is_utility_vehicle"] = safe_equals(
        df,
        "VehicleCategory",
        "Utility"
    )

    df["is_sport_vehicle"] = safe_equals(
        df,
        "VehicleCategory",
        "Sport"
    )

    df["has_multiple_cars"] = safe_not_equals(
        df,
        "NumberOfCars",
        "1 vehicle"
    )

    return df


# =========================================================
# Step 9: Policy features
# =========================================================

@register_feature_step(
    name="policy_features",
    description="Creates policy type, base policy and deductible-related features."
)
def add_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_all_perils_policy"] = safe_equals(
        df,
        "BasePolicy",
        "All Perils"
    )

    df["is_collision_policy"] = safe_equals(
        df,
        "BasePolicy",
        "Collision"
    )

    df["is_liability_policy"] = safe_equals(
        df,
        "BasePolicy",
        "Liability"
    )

    df["is_high_deductible"] = safe_numeric_greater_equal(
        df,
        "Deductible",
        500
    )

    return df


# =========================================================
# Step 10: Demographic features
# =========================================================

@register_feature_step(
    name="demographic_features",
    description="Creates age and gender-related features."
)
def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_young_driver"] = safe_numeric_less_than(
        df,
        "Age",
        25
    )

    if column_exists(df, "Age"):
        age_values = pd.to_numeric(df["Age"], errors="coerce")
        df["is_senior_driver"] = (age_values >= 65).astype(int)
    else:
        df["is_senior_driver"] = 0

    df["is_male"] = safe_equals(df, "Sex", "Male")
    df["is_female"] = safe_equals(df, "Sex", "Female")

    return df


# =========================================================
# Main pipeline
# =========================================================

def apply_feature_engineering(
    df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply all registered feature engineering steps.

    This function is used both during training and prediction.

    Important:
    - It does not split train/test.
    - It does not fit encoders or imputers.
    - It only creates or cleans features.
    """

    df = df.copy()

    original_columns = set(df.columns)

    for step in FEATURE_ENGINEERING_STEPS:
        step_name = step["name"]
        step_function = step["function"]

        before_columns = set(df.columns)
        df = step_function(df)
        after_columns = set(df.columns)

        new_columns = sorted(list(after_columns - before_columns))

        if verbose:
            print(f"[FEATURE ENGINEERING] Step: {step_name}")
            print(f"New columns added: {new_columns}")
            print("-" * 60)

    final_columns = set(df.columns)
    all_new_columns = sorted(list(final_columns - original_columns))

    if verbose:
        print("=" * 70)
        print("FEATURE ENGINEERING COMPLETED")
        print("=" * 70)
        print(f"Original column count: {len(original_columns)}")
        print(f"Final column count   : {len(final_columns)}")
        print(f"New feature count    : {len(all_new_columns)}")
        print("\nNew features:")
        for col in all_new_columns:
            print(f"- {col}")

    return df


def get_feature_engineering_steps() -> pd.DataFrame:
    """
    Return registered feature engineering steps as a dataframe.
    Useful for documentation, debugging and reporting.
    """

    rows = []

    for order, step in enumerate(FEATURE_ENGINEERING_STEPS, start=1):
        rows.append({
            "order": order,
            "name": step["name"],
            "description": step["description"],
            "function": step["function"].__name__
        })

    return pd.DataFrame(rows)


def get_created_features(df: pd.DataFrame) -> list:
    """
    Compare original dataframe columns with engineered dataframe columns
    and return the list of newly created features.
    """

    original_columns = set(df.columns)
    engineered_df = apply_feature_engineering(df, verbose=False)
    engineered_columns = set(engineered_df.columns)

    return sorted(list(engineered_columns - original_columns))


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":
    from config import DATA_PATH

    raw_df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("REGISTERED FEATURE ENGINEERING STEPS")
    print("=" * 70)
    print(get_feature_engineering_steps())

    engineered_df = apply_feature_engineering(raw_df, verbose=True)

    print("\nEngineered dataframe shape:")
    print(engineered_df.shape)