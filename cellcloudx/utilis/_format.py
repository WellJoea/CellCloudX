def pv2trim(mesh):
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

def trim2pv(mesh):
    import pyvista as pv
    return pv.wrap(mesh)

def mesh_voxelize(mesh, density= None, check_surface=True):
    import pyvista as pv
    if mesh.__class__.__name__ == "Trimesh":
        mesh = pv2trim(mesh)
    if density is None:
        density = mesh.length/100
    return pv.voxelize(mesh, density=density, check_surface=check_surface)