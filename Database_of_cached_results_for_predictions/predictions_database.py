class Database:
    def __init__(self) -> None:
        pass
    
    def create_database(self):
        pass
    
    def database_already_exists(self) -> bool:
        pass
    
    def insert_new_records_in_database_by_config(self, config_hash, config_json) -> bool:
        pass
    
    def full_reset_database(self) -> bool:
        pass
    
    def reset_by_config(self, config_hash, config_json) -> bool:
        pass
    
    def convert_config_json_into_hash(self) -> int:
        pass
    
    def is_config_already_exists_in_database(self) -> bool:
        pass
    
    def clean_old_records_from_tables(self, save_n_last_days) -> bool:
        pass
    
    def todays_date(self):
        pass