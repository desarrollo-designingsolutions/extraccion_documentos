import asyncio
from sqlalchemy import inspect, text
from database import engine, Base

async def run_migrations():
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.commit()

        # Usar run_sync para métodos sync como inspect
        async def get_tables(conn):
            return inspect(conn).get_table_names()

        tablas_existentes = await conn.run_sync(get_tables)

        if not tablas_existentes:
            # create_all es sync, así que usa run_sync
            await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    asyncio.run(run_migrations())  # Ejecuta async si se llama desde main
    print("🎯 Migraciones completadas!")