# Dataset base:
# Dataset base (usar en todos los ejercicios):
import pandas as pd
data = {
"Producto": ["Laptop","Laptop","Mouse","Mouse","Monitor","Monitor","Teclado","Teclado"],
"Marca": ["Dell","HP","Logitech","Razer","Samsung","LG","Logitech","Razer"],
"Categoria": ["Computadoras","Computadoras","Accesorios","Accesorios","Monitores","Monitores","Accesorios","Accesorios"],
"Precio": [900,850,25,40,300,280,45,60],
"Unidades_Vendidas": [10,15,50,30,20,18,35,25]
}
df = pd.DataFrame(data)

print(df)
# 1. Mostrar las unidades vendidas totales por producto usando groupby.

print(df.groupby(by="Producto").aggregate(
    {
        "Unidades_Vendidas" : "sum"
    }
))
# 2. Mostrar las unidades vendidas totales por categoría.
print(df.groupby(by="Categoria").aggregate(
    {
        "Unidades_Vendidas" : "sum"
    }
))

# 3. Crear una tabla pivot donde las filas sean Producto, las columnas Categoria y el valor la suma de Unidades_Vendidas.
pivot = df.pivot_table(index="Producto",columns="Categoria",fill_value=0 ,aggfunc={"Unidades_Vendidas": "sum"})
print(pivot)
# 4. Crear una tabla pivot donde las filas sean Marca y las columnas Producto.
pivot = df.pivot(index="Marca",columns="Producto")
print(pivot)
# 5. Convertir el DataFrame en un índice multinivel usando Producto y Marca.
pivot = df.set_index(keys=["Producto","Marca"])
print(pivot)
# 6. Usar stack() para convertir las columnas en un nivel adicional del índice.
stack = (df.stack())
# 7. Usar unstack() para volver al formato anterior.
print(stack.unstack())
# 8. Crear una tabla donde las filas sean Categoria, las columnas Marca y el valor la suma de Unidades_Vendidas.
pivot = df.pivot_table(index="Categoria",columns="Marca",fill_value=0 ,aggfunc={"Unidades_Vendidas": "sum"})
print(pivot)
# 9. Mostrar el precio medio por categoría.
print(df.pivot_table(columns="Categoria",aggfunc={
    "Precio":"mean"
}))
# 10. Crear una tabla pivot donde el índice sea Categoria y Producto, las columnas Marca y el valor la suma de Unidades_Vendidas.

pivot = df.pivot_table(index=["Categoria","Producto"],columns="Marca",aggfunc={"Unidades_Vendidas": "sum"},fill_value=0)
print(pivot)