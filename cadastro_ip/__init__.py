"""
Pipeline de tratamento e diagnóstico de cadastros municipais de iluminação pública.

Segue as regras descritas em `agente_ip_instrucoes_v1.4.md`.

Entry point principal:
    from cadastro_ip import pipeline
    resultado = pipeline.executar(cadastro=..., inspecao=..., iae=..., id_=..., municipio=..., uf=...)
"""

__version__ = "1.4.0"
