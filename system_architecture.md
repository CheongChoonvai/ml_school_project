# Healthcare Analytics System Architecture

```mermaid
graph TB
    subgraph Data_Sources[Data Sources]
        EHR[Electronic Health Records]
        LAB[Laboratory Systems]
        IMG[Medical Imaging]
        MON[Patient Monitoring Devices]
    end

    subgraph Data_Layer[Data Layer]
        direction TB
        PSQL[(PostgreSQL<br/>Structured Data)]
        MONG[(MongoDB<br/>Unstructured Data)]
        CASS[(Cassandra<br/>Time-series Data)]
        S3[S3/Azure Storage<br/>Large Files]
        REDIS[(Redis Cache)]
    end

    subgraph Processing_Layer[Processing Layer]
        direction TB
        ETL[Data ETL Pipeline]
        PREP[Data Preprocessing]
        FE[Feature Engineering]
        VAL[Data Validation]
    end

    subgraph ML_Layer[ML Layer]
        direction TB
        LOCAL[Local Training]
        CLOUD[Cloud Training]
        subgraph Models[ML Models]
            RF[Random Forest]
            XGB[XGBoost]
            DL[Deep Learning]
        end
        MLOPS[MLOps Pipeline]
    end

    subgraph API_Layer[API Layer]
        REST[REST APIs]
        WS[WebSocket]
        AUTH[Authentication]
        RATE[Rate Limiting]
    end

    subgraph Frontend_Layer[Frontend Layer]
        direction TB
        FLUTTER[Flutter Apps]
        subgraph Components[UI Components]
            DASH[Dashboards]
            ANALYTICS[Analytics Views]
            MONITOR[Patient Monitoring]
        end
    end

    subgraph Security_Layer[Security & Compliance]
        HIPAA[HIPAA Compliance]
        ENCRYPT[Encryption]
        ACCESS[Access Control]
        AUDIT[Audit Logging]
    end

    %% Data flow connections
    EHR --> ETL
    LAB --> ETL
    IMG --> ETL
    MON --> ETL

    ETL --> PSQL
    ETL --> MONG
    ETL --> CASS
    ETL --> S3

    PSQL --> PREP
    MONG --> PREP
    CASS --> PREP
    S3 --> PREP

    PREP --> FE
    FE --> VAL
    VAL --> LOCAL
    VAL --> CLOUD

    LOCAL --> Models
    CLOUD --> Models
    Models --> MLOPS

    MLOPS --> REST
    MLOPS --> WS
    
    REST --> FLUTTER
    WS --> FLUTTER

    %% Security connections
    Security_Layer -.-> Data_Layer
    Security_Layer -.-> Processing_Layer
    Security_Layer -.-> ML_Layer
    Security_Layer -.-> API_Layer
    Security_Layer -.-> Frontend_Layer

    %% Cache connections
    REDIS -.-> API_Layer

    %% Styling
    classDef database fill:#f9f,stroke:#333,stroke-width:2px
    classDef security fill:#f96,stroke:#333,stroke-width:2px
    classDef frontend fill:#9f9,stroke:#333,stroke-width:2px
    classDef processing fill:#99f,stroke:#333,stroke-width:2px
    classDef ml fill:#ff9,stroke:#333,stroke-width:2px

    class PSQL,MONG,CASS,REDIS database
    class HIPAA,ENCRYPT,ACCESS,AUDIT security
    class FLUTTER,DASH,ANALYTICS,MONITOR frontend
    class ETL,PREP,FE,VAL processing
    class LOCAL,CLOUD,RF,XGB,DL,MLOPS ml
```

## Architecture Overview

### 1. Data Sources Layer
- Electronic Health Records (EHR)
- Laboratory Systems
- Medical Imaging
- Patient Monitoring Devices

### 2. Data Layer
- PostgreSQL: Structured patient data
- MongoDB: Unstructured medical data
- Cassandra: Time-series monitoring data
- S3/Azure Storage: Large files and images
- Redis: Caching and real-time processing

### 3. Processing Layer
- Data ETL Pipeline
- Data Preprocessing
- Feature Engineering
- Data Validation

### 4. ML Layer
- Local Training Pipeline
- Cloud Training Pipeline
- Model Types:
  - Random Forest
  - XGBoost
  - Deep Learning
- MLOps Pipeline

### 5. API Layer
- REST APIs
- WebSocket Connections
- Authentication/Authorization
- Rate Limiting

### 6. Frontend Layer (Flutter)
- Dashboards
- Analytics Views
- Patient Monitoring
- Real-time Updates

### 7. Security & Compliance Layer
- HIPAA Compliance
- Data Encryption
- Access Control
- Audit Logging

## Data Flow Description

1. **Data Ingestion**
   - Multiple data sources feed into ETL pipeline
   - Data is validated and routed to appropriate storage

2. **Data Processing**
   - Raw data is preprocessed
   - Features are engineered
   - Data is validated for quality

3. **Model Training**
   - Training occurs both locally and in cloud
   - Models are versioned and tracked
   - Performance is monitored

4. **API Integration**
   - Models serve predictions via APIs
   - Real-time updates via WebSocket
   - Secured and rate-limited access

5. **Frontend Delivery**
   - Flutter apps consume APIs
   - Real-time updates displayed
   - Interactive visualizations

6. **Security Overlay**
   - All layers protected by security measures
   - Continuous compliance monitoring
   - Comprehensive audit logging
