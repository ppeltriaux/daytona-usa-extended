from dtools.arcade.polyrom import Poly
from dtools.geom.obj import write_obj

N = (0.0, 1.0, 0.0)


def test_groups_vertices_faces_and_running_indices(tmp_path):
    quad = Poly(((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), N, 0x201)
    tri = Poly(((0, 0, 0), (5, 0, 0), (5, 1, 0)), N, 0x301)
    p = tmp_path / "a.obj"
    write_obj(p, {"road": [quad], "tree": [tri]})
    lines = p.read_text().splitlines()
    assert "g road" in lines
    assert sum(l.startswith("v ") for l in lines) == 7
    faces = [l for l in lines if l.startswith("f ")]
    assert len(faces) == 2
    assert faces[0] == "f 1 2 3 4"
    assert faces[1].split()[1:] == ["5", "6", "7"]
    assert "# attr 201" in lines and lines.index("# attr 201") < lines.index("f 1 2 3 4")
