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

// Helper Function to make a map of node ids to 3d coordinates
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

// My template rssi function it is the inverse of y = a log (bx + 1)
const rssiFunction = (a: number, b: number) => {
    return (rssi: number) => (Math.pow(10, rssi / (-1 * a)) + 1) / b;
};

const main = async () => {
    // list of tag ids from database
    let tags: string[];
    try {
        const tagsResponse: tags[] = await prisma.tags.findMany();
        tags = tagsResponse.map((tag) => tag.id);
    } catch (err) {
        console.error(`RETRIVAL OF TAG DATA FAILED\nERROR: ${err}`);
        process.exit(-1);
    }

    // list of nodes from database
    let nodes: nodes[] = [];
    try {
        nodes = await prisma.nodes.findMany();
    } catch (err) {
        console.error(`RETRIVAL OF TAG DATA FAILED\nERROR: ${err}`);
        process.exit(-1);
    }
    // nodeMap is just a lookup table of node ID -> 3d coordinates
    const nodeMap = mapNodes(nodes);

    const queryResult = await prisma.computed_data.aggregate({
        _max: {
            timestamp: true,
        },
    });
    const latestComputedDataTimestamp = queryResult._max.timestamp ?? 0;

    // next block just keeps the ids from conflicting by taking the largest
    // ID in database
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

    // deal with data one tag at a time
    for (const tag of tags) {
        const tagId = tag;

        // an evaluation is just a term I use to designate a group of
        // node ping entries who are close together in timestamp
        const evaluations = new Map<number, node_ping_entries[]>();

        let entries: node_ping_entries[];
        try {
            entries = await prisma.node_ping_entries.findMany({
                where: {
                    tag_id: tagId,
                    timestamp: {
                        gt: latestComputedDataTimestamp,
                    },
                },
                orderBy: {
                    timestamp: "asc",
                },
            });
        } catch (err) {
            console.error(
                `RETRIVAL OF ENTRY DATA FAILED FOR TAG ${tagId}\nERROR: ${err}`
            );
            process.exit(-1);
        }

        entries.forEach((entry) => {
            let time = entry.timestamp;
            // the entries are grouped into 4 sec intervals
            // ping rate of the tags is every 15 seconds and this should be
            // more than enough of a tolerance gap for the entries
            time -= time % 4;
            const evalArr = evaluations.get(time) ?? [];
            evalArr.push(entry);
            evaluations.set(time, evalArr);
        });

        // these variables are temporary ones to deal with
        // any pings with duplicate data and or
        // not enough data
        const deletions: number[] = [];
        const updateMap = new Map<number, node_ping_entries[]>();

        evaluations.forEach((val, key) => {
            // this lovely mess will first make a list of any duplicate nodes
            // and then will filter out the duplicates past the first
            const newVal = val.filter((val, i, arr) => {
                return (
                    arr.filter((val2, index2) => {
                        return val.node_id == val2.node_id && index2 > i;
                    }).length === 0
                );
            });

            // trilateration works like shit without 3 points lol
            // actually technically 4 are needed to be technically
            // accurate but this is our secret
            // if there aren't 3 points add to the deletion que
            // and if we had duplicates we filtered out add it to
            // the update que
            if (newVal.length < 3) {
                deletions.push(key);
            } else if (newVal.length != val.length) {
                updateMap.set(key, newVal);
            }
        });

        // these process updates and deletions from the map
        deletions.forEach((key) => evaluations.delete(key));
        updateMap.forEach((val, key) => {
            evaluations.set(key, val);
        });

        // loop to generate the data
        for (const evaluation of evaluations.entries()) {
            // the evalArr contains every node entry related to the timestamp
            const [timestamp, evalArr] = evaluation;

            // avg of the positions will be generated
            // as starting point for the
            // trilateration formula
            let avg: number[] = [0, 0, 0];
            let avgCount = 0;

            const distances: number[] = [];
            const points: Vector3d[] = [];

            // I include this indices array so that I can have an input
            // array that for the mean least squares formula
            // input will be an index relating to distance and point data
            // output will always be 0 as we are trying to find the least incorrect
            // guess
            const indices: number[] = [];

            const relationData: data_and_node_relations[] = [];

            for (const evalEntry of evalArr) {
                // These A and B numbers were generated using another script of mine
                // at https://github.com/clupnyluke/rssi-regression-calc
                // they are fitting a small horrible set of data trying to relate distance
                // and rssi through a lot of impossible real world factors
                // margin of error is massive
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

                // this is going to be used to relate the nodes used in calculation to the
                // data itself
                relationData.push({
                    id: relationIdCount++,
                    timestamp: timestamp,
                    entry_id: evalEntry.id,
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
                // Don't bother wondering how I picked my conditions for the LM funtion
                // Because I also wonder how I picked these conditions
                // Frankly it was guess and see what numbers I liked
                // I do like the results tho so I hope I did things right
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
                prisma.computed_data.create({
                    data: {
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
                    prisma.data_and_node_relations.createMany({
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
