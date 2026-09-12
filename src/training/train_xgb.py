import polars as pl
from xgboost import XGBClassifier


FEATURES = [
    "general_score",
    "general_recency_score",
    "type_score",
    "time_score",
    "buy_score",
    "event_count",
    "click_count",
    "cart_count",
    "order_count",
    "last_position",
    "was_clicked",
    "was_carted",
    "was_ordered",
    "time_since_last_interaction",
    "global_event_count",
    "global_click_count",
    "global_cart_count",
    "global_order_count",
    "unique_sessions",
    "session_event_count",
    "session_unique_items",
    "session_click_count",
    "session_cart_count",
    "session_order_count",
    "last_position_ratio",
    "item_cart_rate",
    "item_order_rate",
    "session_repeat_ratio"
]


def prepare_training_data(train_df: pl.DataFrame, val_df: pl.DataFrame, label_column: str = "click_label"):
    X_train = train_df.select(FEATURES).to_pandas()
    y_train = train_df[label_column].to_numpy()

    X_val = val_df.select(FEATURES).to_pandas()
    y_val = val_df[label_column].to_numpy()

    print("Train rows:", len(train_df))
    print(f"Positive {label_column}:", y_train.sum())
    print(f"{label_column} ratio:", y_train.mean())

    return (X_train, y_train), (X_val, y_val)


def train_binary_model(X_train, y_train, X_val, y_val):
    positive_count = y_train.sum()
    if positive_count == 0:
        raise ValueError("The training fold has no positive labels")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - positive_count) / positive_count,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    return model
