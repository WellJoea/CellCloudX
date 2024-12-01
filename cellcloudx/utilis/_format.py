def pv2trimm(mesh):
    import trimesh
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    else:
        import pyvista as pv
        if isinstance(mesh, pv.PolyData):
            tmesh = mesh.extract_surface().triangulate()
            faces_as_array = tmesh.faces.reshape((tmesh.n_cells, 4))[:, 1:]
            tmesh = trimesh.Trimesh(tmesh.points, faces_as_array) 
            return tmesh
