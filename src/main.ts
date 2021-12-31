import latlonEllipsoidal, {
    Vector3d,
    Cartesian,
} from "geodesy/latlon-ellipsoidal";
import {
    data_and_node_relations,
    nodes,
    node_ping_entries,
    PrismaClient,
    tags,
} from ".prisma/client";
import LM from "ml-levenberg-marquardt";

const prisma = new PrismaClient();

const mapNodes = (nodes: nodes[]) => {
    const map = new Map<string, Vector3d>();
    for (const node of nodes) {
        map.set(
            node.id,
            new latlonEllipsoidal(
                node.latitiude,
                node.longitude,
                159
            ).toCartesian()
        );
    }
    return map;
};

const rssiFunction = (a: number, b: number) => {
    return (rssi: number) => (Math.pow(10, rssi / (-1 * a)) + 1) / b;
};

const main = async () => {
    let tags: string[];
    try {
        const tagsResponse: tags[] = await prisma.tags.findMany(); // probably need to wait to map
        tags = tagsResponse.map((tag) => tag.id);
    } catch (err) {
        console.error(`RETRIVAL OF TAG DATA FAILED\nERROR: ${err}`);
        process.exit(-1);
    }

    let nodes: nodes[] = [];
    try {
        nodes = await prisma.nodes.findMany();
    } catch (err) {
        console.error(`RETRIVAL OF TAG DATA FAILED\nERROR: ${err}`);
        process.exit(-1);
    }
    const nodeMap = mapNodes(nodes);

    let dataIdCount = 0;
    try {
        const currentDataIdCount = await prisma.computed_data.aggregate({
            _max: {
                id: true,
            },
        });
        dataIdCount = currentDataIdCount._max.id ?? 0;
    } catch (err) {
        console.error(`RETRIEVAL OF MAX DATA ID FAILED\nERROR: ${err}`);
    }

    let relationIdCount = 0;
    try {
        const currentRelationIdCount =
            await prisma.data_and_node_relations.aggregate({
                _max: {
                    id: true,
                },
            });
        relationIdCount = currentRelationIdCount._max.id ?? 0;
    } catch (err) {
        console.error(
            `RETRIEVAL OF MAX RELATION DATA ID FAILED\nERROR: ${err}`
        );
    }

    for (const tag of tags) {
        const evaluations = new Map<number, node_ping_entries[]>();
        const tagId = tag;

        let entries: node_ping_entries[];
        try {
            entries = await prisma.node_ping_entries.findMany({
                where: {
                    tag_id: tagId,
                },
                orderBy: {
                    timestamp: "asc",
                },
                take: 100,
            });
        } catch (err) {
            console.error(
                `RETRIVAL OF ENTRY DATA FAILED FOR TAG ${tagId}\nERROR: ${err}`
            );
            process.exit(-1);
        }

        entries.forEach((entry) => {
            let time = entry.timestamp;
            time -= time % 4;
            const evalArr = evaluations.get(time) ?? [];
            evalArr.push(entry);
            evaluations.set(time, evalArr);
        });

        const deletions: number[] = [];
        const updateMap = new Map<number, node_ping_entries[]>();

        evaluations.forEach((val, key) => {
            const newVal = val.filter((val, i, arr) => {
                return (
                    arr.filter((val2, index2) => {
                        return val.node_id == val2.node_id && index2 > i;
                    }).length === 0
                );
            });
            if (newVal.length < 3) {
                deletions.push(key);
            } else if (newVal.length != val.length) {
                updateMap.set(key, newVal);
            }
        });

        deletions.forEach((key) => evaluations.delete(key));
        updateMap.forEach((val, key) => {
            evaluations.set(key, val);
        });

        for (const evaluation of evaluations.entries()) {
            const [timestamp, evalArr] = evaluation;

            let avg: number[] = [0, 0, 0];
            let avgCount = 0;

            const distances: number[] = [];
            const points: Vector3d[] = [];
            const indices: number[] = [];

            const relationData: data_and_node_relations[] = [];

            for (const evalEntry of evalArr) {
                const newDist = rssiFunction(
                    70.6782179721831,
                    0.09839014213440837
                )(evalEntry.signal_strength__rssi_);
                distances.push(newDist);

                const newPoint =
                    nodeMap.get(evalEntry.node_id) ?? new Vector3d(0, 0, 0);
                points.push(newPoint);

                indices.push(avgCount++);
                avg[0] += newPoint.x;
                avg[1] += newPoint.y;
                avg[2] += newPoint.z;

                relationData.push({
                    id: relationIdCount++,
                    node_id: evalEntry.node_id,
                    data_id: dataIdCount,
                });
            }

            avg = avg.map((val) => val / avgCount);

            const fitFunction = ([x, y, z]: number[]) => {
                return () => {
                    let sum = 0;
                    points.forEach((val, index) => {
                        sum += Math.pow(val.x - x, 2);
                        sum += Math.pow(val.y - y, 2);
                        sum += Math.pow(val.z - z, 2);
                        sum -= Math.pow(distances[index], 2);
                    });
                    return sum;
                };
            };

            const result = LM(
                { x: indices, y: new Array(points.length).fill(0) },
                fitFunction,
                {
                    initialValues: avg,
                    maxIterations: 1e5,
                    damping: 1e-25,
                    gradientDifference: 1e-11,
                }
            );

            const cartesian = new Cartesian(
                result.parameterValues[0],
                result.parameterValues[1],
                result.parameterValues[2]
            ).toLatLon();

            let success = true;
            try {
                await prisma.computed_data.create({
                    data: {
                        id: dataIdCount++,
                        timestamp,
                        latititude: cartesian.lat,
                        longitude: cartesian.lon,
                        tag_id: tag as string,
                        possible_depth: 159 - cartesian.height,
                    },
                });
            } catch (err) {
                success = false;
                console.error(
                    `CREATION OF COMPUTED DATA ENTRY FAILED AT: ${timestamp}\nERROR: ${err}`
                );
            }

            if (success) {
                try {
                    await prisma.data_and_node_relations.createMany({
                        data: relationData,
                    });
                } catch (err) {
                    console.error(
                        `CREATION OF DATA RELATION ENTRY FAILED AT: ${timestamp}\nERROR: ${err}`
                    );
                }
            }
        }
    }
};

main();
