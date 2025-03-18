from app.model import register

# since register is used as helper by main() to register all models, we prevent pytest from collecting them
# also added under tox.ini's omit list
register.register_all_models.__test__ = False
