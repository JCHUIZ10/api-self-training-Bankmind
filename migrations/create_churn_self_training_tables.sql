-- ============================================================
-- Tablas de Model Registry & Audit para el módulo CHURN
-- Ejecutar en: BankMindBetta_V3 (o la BD que use el entorno)
-- ============================================================

-- 1. churn_models: metadata de cada modelo entrenado
CREATE TABLE IF NOT EXISTS public.churn_models (
    id_model             BIGSERIAL    PRIMARY KEY,
    model_version        VARCHAR(50)  NOT NULL UNIQUE,
    dagshub_url          VARCHAR(500),
    file_path            VARCHAR(255),
    model_size_mb        NUMERIC(10, 2),
    algorithm            VARCHAR(50)  NOT NULL,
    model_config         JSONB,
    threshold            NUMERIC(5,  4),
    promotion_status     VARCHAR(20)  NOT NULL DEFAULT 'CHALLENGER',
    is_active            BOOLEAN      DEFAULT FALSE,
    predecessor_model_id BIGINT       REFERENCES public.churn_models(id_model) ON DELETE SET NULL,
    created_at           TIMESTAMP    DEFAULT NOW(),
    promoted_at          TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_churn_models_is_active  ON public.churn_models (is_active);
CREATE INDEX IF NOT EXISTS idx_churn_models_status     ON public.churn_models (promotion_status);
CREATE INDEX IF NOT EXISTS idx_churn_models_created    ON public.churn_models (created_at);

-- 2. dataset_churn_prediction: info del dataset usado en cada entrenamiento
CREATE TABLE IF NOT EXISTS public.dataset_churn_prediction (
    id_dataset    BIGSERIAL  PRIMARY KEY,
    total_samples INTEGER,
    count_train   INTEGER,
    count_test    INTEGER,
    churn_ratio   NUMERIC(5, 4),
    smote_applied BOOLEAN    DEFAULT TRUE,
    created_at    TIMESTAMP  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_churn_created ON public.dataset_churn_prediction (created_at);

-- 3. self_training_audit_churn: auditoría completa de cada ciclo de entrenamiento
CREATE TABLE IF NOT EXISTS public.self_training_audit_churn (
    id_audit                  BIGSERIAL   PRIMARY KEY,

    -- Relaciones
    id_dataset                BIGINT      NOT NULL REFERENCES public.dataset_churn_prediction(id_dataset) ON DELETE RESTRICT,
    id_model                  BIGINT      NOT NULL REFERENCES public.churn_models(id_model) ON DELETE RESTRICT,
    id_champion_model         BIGINT      REFERENCES public.churn_models(id_model) ON DELETE SET NULL,

    -- Timing
    start_training            TIMESTAMP,
    end_training              TIMESTAMP,
    training_duration_seconds INTEGER,

    -- Métricas del CHALLENGER
    accuracy                  NUMERIC(6, 5),
    precision_score           NUMERIC(6, 5),
    recall_score              NUMERIC(6, 5),
    f1_score                  NUMERIC(6, 5),
    auc_roc                   NUMERIC(6, 5),

    -- Métricas del CHAMPION (para comparación)
    champion_f1_score         NUMERIC(6, 5),
    champion_recall           NUMERIC(6, 5),
    champion_auc_roc          NUMERIC(6, 5),

    -- Hiperparámetros ganadores
    best_params               JSONB,

    -- Decisión Champion/Challenger
    promotion_status          VARCHAR(20) NOT NULL,
    promotion_reason          TEXT,

    -- Trigger info
    triggered_by              VARCHAR(50),
    trigger_details           JSONB,

    -- Estado
    is_success                BOOLEAN     DEFAULT TRUE,
    error_message             TEXT,

    -- Constraint de fechas
    CONSTRAINT chk_churn_training_dates
        CHECK (end_training IS NULL OR start_training IS NULL OR end_training >= start_training)
);

CREATE INDEX IF NOT EXISTS idx_audit_churn_model   ON public.self_training_audit_churn (id_model);
CREATE INDEX IF NOT EXISTS idx_audit_churn_dataset ON public.self_training_audit_churn (id_dataset);
CREATE INDEX IF NOT EXISTS idx_audit_churn_success ON public.self_training_audit_churn (is_success);
CREATE INDEX IF NOT EXISTS idx_audit_churn_created ON public.self_training_audit_churn (start_training);

-- ============================================================
-- VERIFICAR CREACIÓN
-- ============================================================
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('churn_models', 'dataset_churn_prediction', 'self_training_audit_churn')
ORDER BY table_name;
