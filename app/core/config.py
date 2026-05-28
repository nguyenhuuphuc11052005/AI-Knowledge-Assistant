from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Knowledge Assistant"
    
    # Cấu hình PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    POSTGRES_HOST: str = "localhost"

    # Cấu hình Qdrant
    QDRANT_URL: str
    
    # API Keys
    OPENAI_API_KEY: str

    # Tạo URI kết nối tự động từ các biến trên
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Đọc từ file .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Khởi tạo instance dùng chung cho toàn project
settings = Settings()