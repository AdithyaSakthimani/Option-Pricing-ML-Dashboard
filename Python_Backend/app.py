from polygon import RESTClient

client = RESTClient("h1dAj7gHQxtQsl6g_mJZdfxJF4EfsWK8")

aggs = []
for a in client.list_aggs(
    "AAPL",
    1,
    "minute",
    "2022-01-01",
    "2023-02-03",
    limit=50000,
):
    aggs.append(a)

print(aggs)