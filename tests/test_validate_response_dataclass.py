from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic.dataclasses import dataclass

app = FastAPI()


@dataclass
class Item:
    name: str
    price: float | None = None
    owner_ids: list[int] | None = None


@app.get("/items/invalid", response_model=Item)
def get_invalid():
    return {"name": "invalid", "price": "foo"}


@app.get("/items/innerinvalid", response_model=Item)
def get_innerinvalid():
    return {"name": "double invalid", "price": "foo", "owner_ids": ["foo", "bar"]}


@app.get("/items/invalidlist", response_model=list[Item])
def get_invalidlist():
    return [
        {"name": "foo"},
        {"name": "bar", "price": "bar"},
        {"name": "baz", "price": "baz"},
    ]


client = TestClient(app)


def test_invalid():
    response = client.get("/items/invalid")
    assert response.status_code == 500
    assert response.json()["detail"][0]["loc"] == ["response", "price"]


def test_double_invalid():
    response = client.get("/items/innerinvalid")
    assert response.status_code == 500
    locs = [error["loc"] for error in response.json()["detail"]]
    assert ["response", "price"] in locs
    assert ["response", "owner_ids", 0] in locs


def test_invalid_list():
    response = client.get("/items/invalidlist")
    assert response.status_code == 500
    locs = [error["loc"] for error in response.json()["detail"]]
    assert ["response", 1, "price"] in locs
    assert ["response", 2, "price"] in locs
