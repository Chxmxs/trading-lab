# companion/ml/mlflow_compat.py
"""
Monkey-patch MLflow client for older versions missing MlflowClient.list_experiments().
Importing this module is enough to apply the patch.
"""
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from typing import Any

    def _compat_list_experiments(self: "MlflowClient",
                                 view_type: Any = None,
                                 max_results: int = None,
                                 filter_string: str = None,
                                 order_by: Any = None,
                                 page_token: str = None):
        # Older MLflow requires a positive integer
        try:
            if not isinstance(max_results, int) or max_results < 1:
                max_results = 1000
        except Exception:
            max_results = 1000

        # Default view_type to ACTIVE_ONLY if available
        try:
            from mlflow.entities import ViewType
            vt = view_type if view_type is not None else ViewType.ACTIVE_ONLY
        except Exception:
            vt = view_type

        # Call the underlying search_experiments with a tolerant signature
        try:
            return self.search_experiments(
                view_type=vt,
                max_results=max_results,
                filter_string=filter_string,
                order_by=order_by,
                page_token=page_token,
            )
        except TypeError:
            # Older MLflow without order_by param
            return self.search_experiments(
                view_type=vt,
                max_results=max_results,
                filter_string=filter_string,
                page_token=page_token,
            )

    # Only patch if method missing OR clearly not callable in this version
    if not hasattr(MlflowClient, "list_experiments") or not callable(getattr(MlflowClient, "list_experiments")):
        MlflowClient.list_experiments = _compat_list_experiments  # type: ignore[attr-defined]
except Exception:
    # If MLflow isn't available, do nothing—tests that need it will fail elsewhere.
    pass
