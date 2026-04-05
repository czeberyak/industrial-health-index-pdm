def run_pipeline(data_path: str):
    # 1. Feature Engineering
    processor = VibrationProcessor()
    X_df = processor.build_feature_matrix(data_path)
    X = X_df.values

    # 2. Моделирование
    model = DegradationModel()
    model.fit_degradation_vector(X)
    hi = model.calculate_health_index(X)
    weights, _ = model.get_nmf_components(X)

    # 3. Расчет порогов (на первых 100 замерах)
    baseline_mean = np.mean(hi[:100])
    baseline_std = np.std(hi[:100])
    thresholds = {
        'warning': baseline_mean + 3 * baseline_std,
        'alarm': baseline_mean + 10 * baseline_std
    }

    # 4. Dashboard
    plot_industrial_dashboard(hi, thresholds, weights)

if __name__ == "__main__":
    # run_pipeline('data/ims_bearing_set_1')
    pass
